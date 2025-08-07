# Copyright 2024-present the HuggingFace Inc. team.
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
import numpy as np
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, GPTQConfig, Trainer, BitsAndBytesConfig
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
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    dataset_split: str = field(default="train[:100000]", metadata={"help": "(`['train', 'test', 'eval']`):"})
    dataset_field: list[str] = field(default=None, metadata={"help": "Fields of dataset input and output."})
    dataloader_num_proc: int = field(default=16, metadata={"help": "Number of processes to load dataset"})
    bits: int = field(default=4, metadata={"help": "Number of bits to quantize the model. Default is 4."})

    dataloader_batch_size: int = field(
        default=3000,
        metadata={
            "help": "batch size to load dataset. To set the batch size for training, you should pass --batch_size argument instead."
        },
    )
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    training_mode: str = field(
        default="lora",
        metadata={
            "help": "Training mode: 'full' for full finetuning, 'lora' for LoRA, 'qalora' for QA-LoRA, 'pissa' for PiSSA, 'corda' for CorDA"
        },
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "The rank of LoRA adapter. Used for lora, qalora, and pissa modes."},
    )
    qalora_group_size: int = field(
        default=32,
        metadata={"help": "Group size for QA-LoRA quantization."},
    )
    pissa_niter: int = field(
        default=4,
        metadata={"help": "Number of iterations for PiSSA initialization."},
    )
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})
    # Keep legacy flags for backwards compatibility
    corda_mode: bool = field(default=None, metadata={"help": "Deprecated: use --training_mode=corda instead"})
    use_qalora: bool = field(default=None, metadata={"help": "Deprecated: use --training_mode=qalora instead"})


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


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


def load_or_quantize_model(
    base_model: str, tokenizer, qalora_group_size=32, bits: int = 4, cache_dir: str = "./quantized_models"
) -> AutoModelForCausalLM:
    """
    Load a pre-quantized model from cache or quantize and cache a new one.
    Automatically detects if the model is already GPTQ-quantized.

    Args:
        base_model: Model identifier or path
        tokenizer: Tokenizer for the model
        qalora_group_size: Group size for quantization (default: 32)
        bits: Bit-width for quantization (default: 4)
        cache_dir: Directory to store quantized models

    Returns:
        The loaded (quantized) model
    """
    # First, check if the model is already GPTQ-quantized by trying to load it
    print(f"Checking if {base_model} is already GPTQ-quantized...")
    try:
        # Try to load the model and check if it has GPTQ quantization
        test_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,  # Some GPTQ models might need this
        )

        # Check if the model has GPTQ quantization attributes
        has_gptq = False
        for module in test_model.modules():
            if hasattr(module, "qweight") or hasattr(module, "qzeros") or "gptq" in str(type(module)).lower():
                has_gptq = True
                break

        if has_gptq:
            print(f"✅ Model {base_model} is already GPTQ-quantized. Using directly.")
            return test_model
        else:
            print(f"Model {base_model} is not GPTQ-quantized. Will quantize it with {bits}-bit.")
            # Clen up the test model to free memory
            del test_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    except Exception as e:
        print(f"Could not load model {base_model} directly: {e}")
        print(f"Will attempt to quantize it with {bits}-bit...")

    # If we get here, the model needs to be quantized
    os.makedirs(cache_dir, exist_ok=True)
    model_id = base_model.replace("/", "_").replace("\\", "_")  # Handle Windows paths too
    quantized_model_path = os.path.join(cache_dir, f"{model_id}_gptq_{bits}bit_groupsize_{qalora_group_size}")

    # Check if we already have a cached quantized version with the exact same settings
    if os.path.exists(quantized_model_path) and os.path.exists(os.path.join(quantized_model_path, "config.json")):
        print(f"Loading pre-quantized model from cache: {quantized_model_path}")
        try:
            return AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto")
        except Exception as e:
            print(f"Failed to load cached model: {e}")
            print("Will re-quantize the model...")
            import shutil

            shutil.rmtree(quantized_model_path)  # Remove corrupted cache

    print(
        f"Quantizing model with {bits}-bit and group size {qalora_group_size}, saving to cache: {quantized_model_path}"
    )

    # Configure GPTQ for first-time quantization with the specified bits
    gptq_config = GPTQConfig(
        bits=bits,  # Use the bits parameter from function arguments
        dataset="c4",
        tokenizer=tokenizer,
        group_size=qalora_group_size,
        desc_act=False,
        sym=False,
    )

    # Load and quantize the model
    print(f"Loading model for {bits}-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, device_map="auto", quantization_config=gptq_config, torch_dtype=torch.float16
    )

    # Save the quantized model to cache
    print(f"Saving {bits}-bit quantized model to {quantized_model_path}")
    model.save_pretrained(quantized_model_path)
    tokenizer.save_pretrained(quantized_model_path)

    print(f"✅ Model quantized to {bits}-bit with group size {qalora_group_size} and cached successfully")
    return model


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

    # --- ADD THIS BLOCK ---
    # Set seed before initializing model.
    print(f"Setting random seed to {script_args.seed}")
    random.seed(script_args.seed)
    np.random.seed(script_args.seed)
    torch.manual_seed(script_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(script_args.seed)
        # --- END OF BLOCK ---
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Validate training mode
    valid_modes = ["full", "lora", "qlora", "qalora", "pissa", "corda", "daniel"]
    if script_args.training_mode not in valid_modes:
        raise ValueError(f"Invalid training_mode '{script_args.training_mode}'. Must be one of: {valid_modes}")

    print(f"🏃 Training mode: {script_args.training_mode.upper()}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        model_max_length=script_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Training mode selection
    if script_args.training_mode == "corda":
        print("🔧 Setting up CorDA training...")
        res_model = transformers.AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(
            res_model, script_args.model_name_or_path, subfolder="corda_init", is_trainable=True
        )

    elif script_args.training_mode == "qalora":
        print("🔧 Setting up QA-LoRA training...")
        model = load_or_quantize_model(
            script_args.model_name_or_path,
            tokenizer,
            qalora_group_size=script_args.qalora_group_size,
            bits=script_args.bits,
        )

        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            use_qalora=True,
            qalora_group_size=script_args.qalora_group_size,
            r=script_args.lora_r,
            lora_alpha=2 * script_args.lora_r,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
        )

        model = get_peft_model(model, lora_config)

    elif script_args.training_mode == "pissa":
        print("🔧 Setting up PiSSA training...")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            device_map="auto",
        )

        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=script_args.lora_r,
            lora_alpha=script_args.lora_r,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            init_lora_weights=f"pissa_niter_{script_args.pissa_niter}",  # PiSSA initialization
        )

        model = get_peft_model(model, lora_config)

    elif script_args.training_mode == "lora":
        print("🔧 Setting up LoRA training...")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=script_args.lora_r,
            lora_alpha=2 * script_args.lora_r,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            init_lora_weights=True,
        )

        # model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(model, lora_config)
    elif script_args.training_mode == "qlora":
        print("🔧 Setting up QLoRA training...")

        # Configure 4-bit quantization for QLoRA
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Load model with 4-bit quantization
        model = transformers.AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Prepare model for k-bit training (essential for QLoRA)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        # Configure LoRA for QLoRA
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=script_args.lora_r,
            lora_alpha=2 * script_args.lora_r,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,  # Slightly higher dropout for QLoRA
            bias="none",
            init_lora_weights=True,
        )

        model = get_peft_model(model, lora_config)

        # Enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

        print(f"✅ QLoRA model loaded with 4-bit quantization")

    elif script_args.training_mode == "pissa_rank_analysis":
        print("🔧 Setting up rank analysis with quantization...")

        # Phase 1: Cache FP32 weights (wie vorher)
        print("Phase 1: Lade das originale FP32-Modell zum Cachen der Gewichte...")
        model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path)
        target_modules = ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=script_args.lora_r,
            lora_alpha=2 * script_args.lora_r,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            init_lora_weights="daniel",  # Use a specific init for rank analysis
        )

        from gptqmodel import GPTQModel
        from gptqmodel.quantization import QuantizeConfig

        peft_model = get_peft_model(model, lora_config)  # Adapter model with LoRA config

        base_model = model.base_model
        base_model.save_pretrained(os.path.join(script_args.output_dir, "ft", "base_model"))

        # add in here different quantization configs for different bits from 1 to 4
        bits = 4
        # Configure GPTQ for first-time quantization with the specified bits
        gptq_config = GPTQConfig(
            bits=script_args.bits,  # Use the bits parameter from function arguments
            dataset="c4",
            tokenizer=tokenizer,
            group_size=32,
            desc_act=False,
            sym=False,
            static_groups=False,  # +2-5% quality
            true_sequential=True,  # +1-3% quality
            actorder=True,  # +3-7% quality
        )

        # Load and quantize the model
        print(f"Loading model for {bits}-bit quantization...")
        quantized_model = AutoModelForCausalLM.from_pretrained(
            os.path.join(script_args.output_dir, "ft", "base_model"), device_map="auto", quantization_config=gptq_config, torch_dtype=torch.float16
        )

        # Step 5: Re-apply PEFT to quantized model
        peft_model.base_model.model = quantized_model
        peft_model.base_model.model = prepare_model_for_kbit_training(
            peft_model.base_model.model,
            use_gradient_checkpointing=True
        )
        print(f"✅ QLoRA model loaded with 4-bit quantization and FP32 weights attached for rank analysis")

    elif script_args.training_mode == "full":
        print("🔧 Setting up full finetuning...")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    elif script_args.training_mode == "daniel_old":
        print("🔧 Setting up QA-LoRA training...")
        model = load_or_quantize_model(
            script_args.model_name_or_path,
            tokenizer,
            qalora_group_size=script_args.qalora_group_size,
            bits=script_args.bits,
        )

        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            use_qalora=True,
            qalora_group_size=script_args.qalora_group_size,
            r=script_args.lora_r,
            lora_alpha=2 * script_args.lora_r,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            init_lora_weights="daniel",  # PiSSA initialization
        )

        model = get_peft_model(model, lora_config)
    elif script_args.training_mode == "daniel":
        print("🔧 Setting up QA-LoRA training with PiSSA initialization...")

        # =================================================================================
        # PHASE 1: Sichern der originalen FP32-Gewichte
        # =================================================================================
        print("Phase 1: Lade das originale FP32-Modell zum Cachen der Gewichte...")
        # Lade das hochpräzise Originalmodell
        fp32_model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path)

        # Definiere die Zielmodule hier, damit wir wissen, welche Gewichte wir speichern müssen
        target_modules = ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]

        # Erstelle eine Map, die Layernamen auf ihre FP32-Gewichte abbildet
        fp32_weights_map = {
            name: module.weight.clone().to(torch.float32)
            for name, module in fp32_model.named_modules()
            # Speichere nur die Gewichte, die wir für PiSSA brauchen werden
            if any(target_module in name for target_module in target_modules)
        }

        print(f"  -> {len(fp32_weights_map)} FP32-Gewichtsmatrizen für Ziel-Layer gecached.")

        # Gib den Speicher des großen FP32-Modells sofort frei
        del fp32_model
        import gc

        gc.collect()
        torch.cuda.empty_cache()

        # =================================================================================
        # PHASE 2: Modell laden/quantisieren und Gewichte anheften
        # =================================================================================
        print("\nPhase 2: Lade oder quantisiere das Hauptmodell...")
        model = load_or_quantize_model(
            script_args.model_name_or_path,
            tokenizer,
            qalora_group_size=script_args.qalora_group_size,
            bits=script_args.bits,
        )
        print("  -> Modell geladen/quantisiert.")

        print("\nPhase 3: Hänge die gecachten FP32-Gewichte an die quantisierten Layer an...")
        # Iteriere durch die Layer des *quantisierten* Modells
        for name, module in model.named_modules():
            if name in fp32_weights_map:
                # Hänge das gecachte FP32-Gewicht als neues Attribut an den Layer an.
                # Ihre `daniel_init` Funktion wird nach diesem Attribut suchen.
                module.original_weight_fp32 = fp32_weights_map[name].to(module.qweight.device)
                print(f"  - Originalgewicht für '{name}' erfolgreich angehängt.")

        # =================================================================================
        # PHASE 3: PEFT-Setup mit PiSSA (Ihr ursprünglicher Code)
        # =================================================================================
        print("\nPhase 4: Richte PEFT mit PiSSA-Initialisierung ein...")
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            use_qalora=True,
            qalora_group_size=script_args.qalora_group_size,
            r=script_args.lora_r,
            lora_alpha=2 * script_args.lora_r,
            target_modules=target_modules,  # Wiederverwendung der oben definierten Liste
            lora_dropout=0.05,
            bias="none",
            init_lora_weights="daniel_niter_5",  # WICHTIG: Stellen Sie sicher, dass Sie hier die Anzahl der Iterationen angeben
        )

        model = get_peft_model(model, lora_config)
        print("  -> PEFT-Modell erfolgreich erstellt. PiSSA-Initialisierung wird beim Trainingsstart ausgelöst.")
    else:
        raise ValueError(f"Unknown training mode: {script_args.training_mode}")

    trainable_params, all_param = get_nb_trainable_parameters(model)
    print(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.2f}%"
    )

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

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = {
        "train_dataset": train_dataset,
        "data_collator": data_collator,
    }
    trainer = Trainer(model=model, tokenizer=tokenizer, args=script_args, **data_module)
    trainer.train()
    trainer.save_state()
    model.save_pretrained(os.path.join(script_args.output_dir, "ft"))
    tokenizer.save_pretrained(os.path.join(script_args.output_dir, "ft"))


if __name__ == "__main__":
    train()
