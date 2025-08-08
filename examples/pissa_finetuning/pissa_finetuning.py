# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer

from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training


IGNORE_INDEX = -100

PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


def get_nb_trainable_parameters(model) -> tuple[int, int]:
    r"""
    Returns the number of trainable parameters and the number of all parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            num_bytes = param.quant_storage.itemsize if hasattr(param, "quant_storage") else 1
            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # Model configs
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    residual_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name or path of the fp32/16 residual model. (`['fxmeng/pissa-llama-2-7b-r16-alpha-16']`)"
        },
    )

    # Dataset configs
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    dataset_split: str = field(default="train[:100000]", metadata={"help": "(`['train', 'test', 'eval']`):"})
    dataset_field: list[str] = field(default=None, metadata={"help": "Fields of dataset input and output."})
    dataloader_num_proc: int = field(default=16, metadata={"help": "Number of processes to load dataset"})
    dataloader_batch_size: int = field(
        default=3000,
        metadata={
            "help": "batch size to load dataset. To set the batch size for training, you should pass --batch_size argument instead."
        },
    )

    # Training configs
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

    # PiSSA specific configs
    bits: str = field(default="bf16", metadata={"help": "(`['fp4', 'nf4', 'int8', 'bf16', 'fp16', fp32]`)"})
    init_lora_weights: str = field(
        default="pissa_niter_4", metadata={"help": "(`['gaussian', 'pissa', 'pissa_niter_4']`)"}
    )
    lora_r: int = field(default=16, metadata={"help": "The rank of LoRA adapter."})
    lora_alpha: int = field(default=16, metadata={"help": "The alpha parameter of LoRA."})
    lora_dropout: float = field(default=0.05, metadata={"help": "The dropout parameter of LoRA."})
    convert_pissa_to_lora: bool = field(default=True, metadata={"help": "Convert PiSSA to LoRA format after training"})
    merge_and_save: bool = field(default=False, metadata={"help": "Merge and save the final model"})


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "input_ids_lens": input_ids_lens,
        "labels_lens": labels_lens,
    }


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = (_tokenize_fn(strings, tokenizer) for strings in (examples, sources))
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return {
        "input_ids": input_ids,
        "labels": labels,
    }


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[dict]) -> dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }


def train_tokenize_function(examples, tokenizer, query, response):
    sources = [
        PROMPT.format_map(
            {
                "instruction": instruction,
            }
        )
        for instruction in examples[query]
    ]
    targets = [f"{output}{tokenizer.eos_token}" for output in examples[response]]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict


def train():
    parser = transformers.HfArgumentParser(TrainingArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    print(script_args)

    print("🏃 Training mode: PiSSA")

    # Load tokenizer
    if script_args.residual_model_name_or_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(script_args.residual_model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.model_max_length = script_args.model_max_length
    tokenizer.padding_side = "right"

    # Model loading logic (same as original)
    print(f"Load pre-processed residual model in {script_args.bits} bits.")

    if script_args.bits in ["nf4", "fp4", "int8"]:
        print("🔧 Setting up PiSSA training with quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=(script_args.bits == "nf4" or script_args.bits == "fp4"),
            load_in_8bit=script_args.bits == "int8",
            bnb_4bit_quant_type=script_args.bits,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        res_model = AutoModelForCausalLM.from_pretrained(
            script_args.residual_model_name_or_path, quantization_config=quantization_config, low_cpu_mem_usage=True
        )
        res_model = prepare_model_for_kbit_training(res_model)
        print("Wrapping the residual model with PiSSA.")
        peft_model = PeftModel.from_pretrained(
            res_model, script_args.residual_model_name_or_path, subfolder="pissa_init", is_trainable=True
        )

    elif script_args.residual_model_name_or_path is not None:
        print("🔧 Setting up PiSSA training with pre-processed model...")
        res_model = AutoModelForCausalLM.from_pretrained(
            script_args.residual_model_name_or_path,
            torch_dtype=(
                torch.float16
                if script_args.bits == "fp16"
                else (torch.bfloat16 if script_args.bits == "bf16" else torch.float32)
            ),
            device_map="auto",
        )
        print("Wrapping the residual model with PiSSA.")
        peft_model = PeftModel.from_pretrained(
            res_model, script_args.residual_model_name_or_path, subfolder="pissa_init", is_trainable=True
        )

    elif script_args.model_name_or_path is not None:
        print("🔧 Setting up PiSSA training from base model...")
        print(f"No available pre-processed model, manually initialize a PiSSA using {script_args.model_name_or_path}.")

        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            torch_dtype=(
                torch.float16
                if script_args.bits == "fp16"
                else (torch.bfloat16 if script_args.bits == "bf16" else torch.float32)
            ),
            device_map="auto",
        )

        lora_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_alpha,
            init_lora_weights=script_args.init_lora_weights,
            lora_dropout=script_args.lora_dropout,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM"
            )
        peft_model = get_peft_model(model, lora_config)

    else:
        raise ValueError("Either model_name_or_path or residual_model_name_or_path must be provided")

    trainable_params, all_param = get_nb_trainable_parameters(peft_model)
    print(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.2f}%"
    )

    # Dataset preparation (same as corda_finetuning.py)
    raw_train_datasets = load_dataset(script_args.data_path, split=script_args.dataset_split)
    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        batch_size=script_args.dataloader_batch_size,
        num_proc=script_args.dataloader_num_proc,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset",
        fn_kwargs={
            "tokenizer": tokenizer,
            "query": script_args.dataset_field[0],
            "response": script_args.dataset_field[1],
        },
    )

    # Training setup (same as corda_finetuning.py)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = {
        "train_dataset": train_dataset,
        "data_collator": data_collator,
    }

    trainer = Trainer(model=peft_model, tokenizer=tokenizer, args=script_args, **data_module)
    trainer.train()
    trainer.save_state()

    # PiSSA-specific saving logic
    if script_args.convert_pissa_to_lora:
        print("Converting PiSSA to LoRA format...")
        if script_args.residual_model_name_or_path is not None:
            peft_model.save_pretrained(
                os.path.join(script_args.output_dir, "pissa_lora"),
                path_initial_model_for_weight_conversion=os.path.join(
                    script_args.residual_model_name_or_path, "pissa_init"
                ),
            )
        else:
            # For base model initialization, save as regular adapter
            peft_model.save_pretrained(os.path.join(script_args.output_dir, "pissa_lora"))
    else:
        peft_model.save_pretrained(os.path.join(script_args.output_dir, "pissa_ft"))

    if script_args.merge_and_save:
        print("Merging and saving final model...")
        model = peft_model.merge_and_unload()
        model.save_pretrained(os.path.join(script_args.output_dir, "pissa_merged"))
        tokenizer.save_pretrained(os.path.join(script_args.output_dir, "pissa_merged"))

    # Always save tokenizer with the adapter
    save_dir = "pissa_lora" if script_args.convert_pissa_to_lora else "pissa_ft"
    tokenizer.save_pretrained(os.path.join(script_args.output_dir, save_dir))


if __name__ == "__main__":
    train()
