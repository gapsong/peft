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
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional, List
import datetime
import numpy as np
import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, GPTQConfig, Trainer
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from safetensors.torch import load_file
from eval_peft import load_model_and_tokenizer
import json

def unpack_qweight2(qweight: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Entpackt einen GPTQ-komprimierten qweight-Tensor in einzelne Integer-Werte.
    """
    if 32 % bits != 0:
        raise NotImplementedError(f"Reines Python-Entpacken für {bits}-Bit wird nicht unterstützt, da 32 nicht durch {bits} teilbar ist.")
    
    pack_factor = 32 // bits
    mask = (1 << bits) - 1

    unpacked = torch.zeros(
        (qweight.shape[0]* pack_factor, qweight.shape[1] ),
        dtype=torch.int8,  # Verwende int8, um sicher zu sein
        device=qweight.device
    )
    for j in range(pack_factor):
        shift = bits * j
        unpacked[j :: pack_factor, :] = (qweight >> shift) & mask

    return unpacked

def unpack_qweight(qweight: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Entpackt einen GPTQ-komprimierten qweight-Tensor basierend auf der originalen GPTQ-Logik.
    Folgt exakt der dequantize_weight Implementierung.
    """
    if bits in [2, 4, 8]:
        # Standard unpacking for 2, 4, 8 bit
        pack_factor = 32 // bits
        maxq = (1 << bits) - 1
        
        # Create shift patterns (equivalent to wf_unsqueeze_neg_one)
        wf = torch.arange(0, pack_factor * bits, bits, dtype=torch.int32, device=qweight.device)
        wf_unsqueeze_neg_one = wf.unsqueeze(-1)  # Shape: (pack_factor, 1)
        
        # Unpack weights following the exact GPTQ logic
        weight = torch.bitwise_and(
            torch.bitwise_right_shift(
                torch.unsqueeze(qweight, 1).expand(-1, pack_factor, -1),
                wf_unsqueeze_neg_one  # self.wf.unsqueeze(-1)
            ).to(torch.int32),
            maxq,
        )
        
    elif bits == 3:
        # Special handling for 3-bit quantization - exact copy from GPTQ
        wf = torch.tensor([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0], dtype=torch.int32, device=qweight.device)
        wf_unsqueeze_neg_one = wf.unsqueeze(-1)
        
        weight = qweight.reshape(
            qweight.shape[0] // 3, 3, 1, qweight.shape[1]
        ).expand(-1, -1, 12, -1)
        
        weight = (weight >> wf_unsqueeze_neg_one) & 0x7  # self.wf.unsqueeze(-1)
        
        # Handle special cases for 3-bit packing
        weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
        weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
        weight = weight & 0x7
        
        weight = torch.cat(
            [weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1
        )
    else:
        raise NotImplementedError(f"Bits {bits} not supported")
    
    # Final reshape to match expected output format
    # weight.shape is currently (groups, pack_factor, out_features) or similar
    # We need to flatten it to (in_features, out_features)
    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
    
    return weight

# filepath: /home/nudel/Documents/peft/examples/qalora_finetuning/corda_finetuning.py
# Replace the existing unpack_qweight function with this one

def unpack_qzeros(qzeros: torch.Tensor, bits: int, group_size: int = 32) -> torch.Tensor:
    """
    Entpackt einen GPTQ-komprimierten qzeros-Tensor basierend auf der originalen GPTQ-Logik.
    """
    if bits in [2, 4, 8]:
        # Standard unpacking for 2, 4, 8 bit
        pack_factor = 32 // bits
        maxq = (1 << bits) - 1
        
        # Create shift patterns (equivalent to wf_unsqueeze_zero)
        wf = torch.arange(0, pack_factor * bits, bits, dtype=torch.int32, device=qzeros.device)
        wf_unsqueeze_zero = wf.unsqueeze(0)  # Shape: (1, pack_factor)
        
        # Expand qzeros and apply bit shifts
        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(qzeros, 2).expand(-1, -1, pack_factor),
            wf_unsqueeze_zero.unsqueeze(0)  # Broadcast to match dimensions
        ).to(torch.int32)
        
        # Apply mask and reshape
        zeros = torch.bitwise_and(zeros, maxq)
        # Reshape to match the expected output format
        zeros = zeros.reshape(qzeros.shape[0], qzeros.shape[1] * pack_factor)
        
    elif bits == 3:
        # Special handling for 3-bit quantization
        wf = torch.tensor([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0], dtype=torch.int32, device=qzeros.device)
        wf_unsqueeze_zero = wf.unsqueeze(0)
        
        zeros = qzeros.reshape(
            qzeros.shape[0], qzeros.shape[1] // 3, 3, 1
        ).expand(-1, -1, -1, 12)
        
        zeros = zeros >> wf_unsqueeze_zero.unsqueeze(0).unsqueeze(0)
        
        # Handle special cases for 3-bit packing
        zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | (
            (zeros[:, :, 1, 0] << 2) & 0x4
        )
        zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | (
            (zeros[:, :, 2, 0] << 1) & 0x6
        )
        zeros = zeros & 0x7
        
        zeros = torch.cat(
            [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]],
            dim=2,
        )
        
        # Reshape to final format
        zeros = zeros.reshape(qzeros.shape[0], -1)
    else:
        raise NotImplementedError(f"Bits {bits} not supported")
    
    return zeros

def pack_qweight(unpacked_qweight: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Packt einen Tensor von Integer-Werten zurück in einen GPTQ-komprimierten qweight-Tensor.
    Basiert auf der originalen GPTQ pack-Funktion.
    """
    # Convert to numpy for bit manipulation (following original implementation)
    int_weight = unpacked_qweight.contiguous().cpu().numpy().astype(np.uint32)
    pack_dtype_bits = 32
    # Calculate pack factor
    if bits in [2, 4, 8]:
        pack_factor = 32 // bits
    elif bits == 3:
        pack_factor = 32 // 3  # This will be handled specially
    else:
        raise ValueError(f"Unsupported bits: {bits}")
    
    if bits in [2, 4, 8]:
        qweight = np.zeros(
            ((int_weight.shape[0] // pack_dtype_bits) * bits, int_weight.shape[1]),
            dtype=np.uint32,
        )
        for row in range(qweight.shape[0]):
            for j in range(pack_factor):
                qweight[row] |= int_weight[row * pack_factor + j] << (bits * j)
                
    elif bits == 3:
        qweight = np.zeros(
            (int_weight.shape[0] // 32 * 3, int_weight.shape[1]),
            dtype=np.uint32,
        )
        i = 0
        row = 0
        while row < qweight.shape[0]:
            for j in range(i, i + 10):
                qweight[row] |= int_weight[j] << (3 * (j - i))
            i += 10
            qweight[row] |= int_weight[i] << 30
            row += 1
            qweight[row] |= (int_weight[i] >> 2) & 1
            i += 1
            for j in range(i, i + 10):
                qweight[row] |= int_weight[j] << (3 * (j - i) + 1)
            i += 10
            qweight[row] |= int_weight[i] << 31
            row += 1
            qweight[row] |= (int_weight[i] >> 1) & 0x3
            i += 1
            for j in range(i, i + 10):
                qweight[row] |= int_weight[j] << (3 * (j - i) + 2)
            i += 10
            row += 1

    return torch.from_numpy(qweight.astype(np.int32)).to(unpacked_qweight.device)

def reconstruct_low_rank_update_from_adapter(adapter_path: str):
    """
    Lädt einen gespeicherten LoRA-Adapter, extrahiert lora_A und lora_B,
    berechnet den Skalierungsfaktor und rekonstruiert die Low-Rank-Update-Matrix.
    """
    # 1. Pfade zur Konfigurations- und Gewichtsdatei definieren
    config_path = os.path.join(adapter_path, "adapter_config.json")
    weights_path = os.path.join(adapter_path, "adapter_model.safetensors")

    if not os.path.exists(config_path) or not os.path.exists(weights_path):
        print(f"❌ Fehler: Konnte 'adapter_config.json' oder 'adapter_model.safetensors' nicht in {adapter_path} finden.")
        return

    # 2. Konfiguration laden, um 'r' und 'lora_alpha' zu erhalten
    print(f"🔍 Lade Konfiguration von: {config_path}")
    with open(config_path, "r") as f:
        adapter_config = json.load(f)

    r = adapter_config.get("r")
    lora_alpha = adapter_config.get("lora_alpha")
    use_rslora = adapter_config.get("use_rslora", False)

    if r is None or lora_alpha is None:
        print("❌ Fehler: 'r' oder 'lora_alpha' nicht in der Konfiguration gefunden.")
        return

    # 3. Den 'scaling'-Parameter berechnen
    if use_rslora:
        scaling = lora_alpha / torch.sqrt(torch.tensor(r))
        print(f"✅ Konfiguration geladen: r={r}, lora_alpha={lora_alpha} (rsLoRA-Skalierung verwendet)")
    else:
        scaling = lora_alpha / r
        print(f"✅ Konfiguration geladen: r={r}, lora_alpha={lora_alpha}")
    
    print(f"   Berechneter Skalierungsfaktor: {scaling:.4f}\n")

    # 4. Das State Dictionary des Adapters laden
    print(f"🔍 Lade Gewichte von: {weights_path}")
    adapter_state_dict = load_file(weights_path)
    print(f"✅ {len(adapter_state_dict)} Tensoren geladen.\n")

    # 5. Durch alle Layer iterieren und die Low-Rank-Matrix rekonstruieren
    reconstructed_matrices = {}
    
    # Finde alle lora_A Gewichte, um die Layer zu identifizieren
    lora_a_keys = [key for key in adapter_state_dict if key.endswith(".lora_A.weight")]

    for lora_a_key in lora_a_keys:
        base_key = lora_a_key.replace(".lora_A.weight", "")
        lora_b_key = base_key + ".lora_B.weight"

        if lora_b_key in adapter_state_dict:
            lora_A = adapter_state_dict[lora_a_key]
            lora_B = adapter_state_dict[lora_b_key]
            
            # Dies ist die Kernlogik Ihrer Anfrage
            # LR = scaling * (B @ A)
            low_rank_update = scaling * (lora_B @ lora_A)
            
            reconstructed_matrices[base_key] = low_rank_update
            
            print(f"--- Rekonstruiert für Layer: {base_key} ---")
            print(f"  - lora_A Form: {lora_A.shape}")
            print(f"  - lora_B Form: {lora_B.shape}")
            print(f"  - Rekonstruierte LR-Matrix Form: {low_rank_update.shape}\n")

    return reconstructed_matrices

def inspect_quantized_safetensors(model_path: str):
    """
    Loads a quantized safetensors file and prints the details of the 
    quantized parameters (qweight, qzeros, scales) for the first found layer.
    """
    # The standard name for the weights file is 'model.safetensors'
    safetensors_file = os.path.join(model_path, "model.safetensors")

    if not os.path.exists(safetensors_file):
        # Fallback for older pytorch .bin format
        safetensors_file = os.path.join(model_path, "pytorch_model.bin")
        if not os.path.exists(safetensors_file):
            print(f"❌ Error: Could not find 'model.safetensors' or 'pytorch_model.bin' in {model_path}")
            return

    print(f"🔍 Loading state dictionary from: {safetensors_file}")
    
    # Use the safetensors library to load the file into a dictionary
    state_dict = load_file(safetensors_file)
    
    print(f"✅ Successfully loaded {len(state_dict)} tensors.\n")
    
    # --- Find and inspect a quantized linear layer ---
    
    found_layer = False
    for key, tensor in state_dict.items():
        # Quantized weights are typically named with a '.qweight' suffix
        if key.endswith(".qweight"):
            print(f"--- Found Quantized Layer: {key.replace('.qweight', '')} ---")
            
            # --- Accessing qweight ---
            qweight_tensor = tensor
            print(f"  - qweight tensor '{key}':")
            print(f"    - Shape: {qweight_tensor.shape}")
            print(f"    - Dtype: {qweight_tensor.dtype}")
            print(f"    - First 5 values: {qweight_tensor.flatten()[:5]}")
            
            # Derive the names of the other related tensors
            base_key = key.replace('.qweight', '')
            qzeros_key = f"{base_key}.qzeros"
            scales_key = f"{base_key}.scales"
            
            # --- Accessing qzeros ---
            if qzeros_key in state_dict:
                qzeros_tensor = state_dict[qzeros_key]
                print(f"\n  - qzeros tensor '{qzeros_key}':")
                print(f"    - Shape: {qzeros_tensor.shape}")
                print(f"    - Dtype: {qzeros_tensor.dtype}")
                print(f"    - First 5 values: {qzeros_tensor.flatten()[:5]}")
            else:
                print(f"\n  - qzeros tensor for {base_key} not found.")

            # --- Accessing scales ---
            if scales_key in state_dict:
                scales_tensor = state_dict[scales_key]
                print(f"\n  - scales tensor '{scales_key}':")
                print(f"    - Shape: {scales_tensor.shape}")
                print(f"    - Dtype: {scales_tensor.dtype}")
                print(f"    - First 5 values: {scales_tensor.flatten()[:5]}")
            else:
                print(f"\n  - scales tensor for {base_key} not found.")

            found_layer = True
            break # Stop after inspecting the first layer

    if not found_layer:
        print("❌ No quantized layers (ending in .qweight) were found in the file.")

IGNORE_INDEX = -100

PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

def find_all_linear_names(model) -> List[str]:
    """
    Finds all linear layer names in a model, excluding the lm_head.
    This is a robust way to automatically select target_modules for LoRA.
    """
    linear_layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Exclude the output layer and embeddings from LoRA training
            if "lm_head" not in name and "embed_tokens" not in name:
                linear_layer_names.append(name)
    print(f"✅ Automatically discovered {len(linear_layer_names)} linear layers to target.")
    return linear_layer_names

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

def compare_models(model1, model2, model1_name="Model 1", model2_name="Model 2", tolerance=1e-9):
    """
    Vergleicht zwei PyTorch-Modelle Parameter für Parameter und gibt detaillierte Unterschiede aus.
    """
    print(f"\n{'='*80}")
    print(f"🔬 Vergleiche '{model1_name}' mit '{model2_name}'")
    print(f"{'='*80}")

    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())

    # 1. Struktureller Vergleich: Haben beide Modelle die gleichen Parameter-Namen?
    keys1 = set(params1.keys())
    keys2 = set(params2.keys())

    if keys1 != keys2:
        print("❌ STRUKTURELLER UNTERSCHIED!")
        missing_in_2 = keys1 - keys2
        missing_in_1 = keys2 - keys1
        if missing_in_2:
            print(f"   - Parameter nur in '{model1_name}' gefunden: {list(missing_in_2)[:5]}")
        if missing_in_1:
            print(f"   - Parameter nur in '{model2_name}' gefunden: {list(missing_in_1)[:5]}")
        return False

    # 2. Detaillierter Vergleich der einzelnen Parameter
    mismatched_params = []
    for name, param1 in params1.items():
        param2 = params2[name]

        # Form-Vergleich
        if param1.shape != param2.shape:
            mismatched_params.append(f"  - '{name}': Form-Mismatch! {param1.shape} vs {param2.shape}")
            continue

        # Datentyp-Vergleich
        if param1.dtype != param2.dtype:
            mismatched_params.append(f"  - '{name}': Dtype-Mismatch! {param1.dtype} vs {param2.dtype}")
            continue

        # Werte-Vergleich
        # torch.equal ist sehr strikt. Wir verwenden allclose für numerische Stabilität.
        if not torch.allclose(param1, param2, atol=tolerance):
            diff = torch.abs(param1 - param2).max().item()
            mismatched_params.append(f"  - '{name}': Werte-Mismatch! Maximale Differenz: {diff:.6e}")
            continue

    # 3. Ergebnis ausgeben
    if not mismatched_params:
        print("✅ Modelle sind identisch!")
        print(f"   - Alle {len(params1)} Parameter stimmen überein.")
        return True
    else:
        print(f"❌ UNTERSCHIEDE GEFUNDEN! ({len(mismatched_params)} von {len(params1)} Parametern)")
        # Gib die ersten 10 Unterschiede aus, um die Konsole nicht zu überfluten
        for mismatch in mismatched_params[:10]:
            print(mismatch)
        if len(mismatched_params) > 10:
            print(f"  ... und {len(mismatched_params) - 10} weitere.")
        return False

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
    valid_modes = ["full", "lora", "qlora", "qalora", "pissa", "corda", "daniel", "pissa_rank_analysis", "daniel_old"]
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
            # torch_dtype=torch.bfloat16,
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
        print("🔧 Setting up rank analysis with multiple quantization configurations...")

        # --- CACHING LOGIC START ---
        # Phase 1: Definiere Pfade und prüfe auf existierende Artefakte
        model_name_clean = script_args.model_name_or_path.replace("/", "_").replace("\\", "_")
        base_output_dir = os.path.join(script_args.output_dir, f"quantized_residuals_r{script_args.lora_r}")
        os.makedirs(base_output_dir, exist_ok=True)

        adapter_name = f"daniel_adapter_r{script_args.lora_r}_{model_name_clean}"
        adapter_path = os.path.join(base_output_dir, adapter_name)
        
        temp_residual_path = os.path.join(script_args.output_dir, f"{model_name_clean}_residual_base_r{script_args.lora_r}_fp16")

        # Prüfe, ob sowohl der Adapter als auch das Residual-Modell bereits existieren
        if os.path.exists(adapter_path) and os.path.exists(temp_residual_path):
            print(f"⏭️  Found cached adapter at: {adapter_path}")
            print(f"⏭️  Found cached residual model at: {temp_residual_path}")
            print("    Skipping model initialization and residual extraction.")
            
            # Lade die target_modules aus der Adapter-Konfiguration für die spätere Verwendung
            from peft import PeftConfig
            try:
                config = PeftConfig.from_pretrained(adapter_path)
                target_modules = config.target_modules
                print(f"    Loaded target_modules from adapter config: {len(target_modules)} modules.")
            except Exception as e:
                print(f"    Could not load adapter config ({e}), using empty target_modules list.")
                target_modules = []
        else:
            print("🔥 No cached artifacts found. Starting full initialization process...")
            
            # Phase 2: Lade das Originalmodell und richte PEFT ein
            print("Phase 2: Loading original model and setting up PEFT...")
            model = AutoModelForCausalLM.from_pretrained(
                script_args.model_name_or_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
            )
            
            # Entdecke automatisch alle relevanten Layer-Namen
            target_modules = find_all_linear_names(model)
            
            # Gib das CPU-Modell frei, wir laden es gleich richtig
            del model
            torch.cuda.empty_cache()

            model = AutoModelForCausalLM.from_pretrained(
                script_args.model_name_or_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )

            lora_config = LoraConfig(
                task_type="CAUSAL_LM",
                r=script_args.lora_r,
                lora_alpha=2 * script_args.lora_r,
                target_modules=target_modules,
                lora_dropout=0.05,
                bias="none",
                init_lora_weights="daniel",
            )

            peft_model = get_peft_model(model, lora_config)
            print("✅ PEFT model with daniel initialization complete")

            # Phase 3: Extrahiere und speichere den Adapter und das Residual-Modell
            print(f"Phase 3: Saving adapter to: {adapter_path}")
            peft_model.save_pretrained(adapter_path)

            print("🚀 Extracting residual weights using State Dict method...")
            peft_base_model = peft_model.get_base_model().to(torch.bfloat16)
            base_state_dict = peft_base_model.state_dict()
            clean_state_dict = {}

            for key, value in base_state_dict.items(): # converts peft_names to hf model names
                if "base_layer.weight" in key:
                    clean_key = key.replace(".base_layer.weight", ".weight")
                    clean_state_dict[clean_key] = value.clone()
                elif "weight" in key and "lora" not in key and "base_layer" not in key:
                    clean_state_dict[key] = value.clone()
            
            # Füge den lm_head manuell hinzu, da get_base_model() ihn oft auslässt
            if "lm_head.weight" in peft_model.state_dict():
                print("🧠 Adding lm_head.weight to the clean state_dict.")
                clean_state_dict["lm_head.weight"] = peft_model.state_dict()["lm_head.weight"].clone()

            print(f"📋 Extracted: {len(clean_state_dict)} clean weights")
            
            residual_model = AutoModelForCausalLM.from_pretrained(
                script_args.model_name_or_path, torch_dtype=torch.bfloat16
            )
            
            residual_model.load_state_dict(clean_state_dict, strict=False)
            
            print(f"💾 Saving clean residual model to: {temp_residual_path}")
            residual_model.save_pretrained(temp_residual_path)
            tokenizer.save_pretrained(temp_residual_path)
            print("✅ Clean residual model saved.")
            
            # Verifiziere den Speicher-/Ladevorgang
            reloaded_model_loaded = AutoModelForCausalLM.from_pretrained(temp_residual_path, torch_dtype=torch.bfloat16)
            compare_models(
                residual_model, reloaded_model_loaded, 
                model1_name="Extracted Residual Model", 
                model2_name="Reloaded Model"
            )

        print("\n🧹 Aggressive Speicherbereinigung vor der Quantisierung...")
        
        # Überprüfe, ob die Variablen existieren, bevor sie gelöscht werden
        # (falls wir aus dem Cache-Pfad kommen, existieren sie nicht)
        if 'model' in locals():
            del model
        if 'peft_base_model' in locals():
            del peft_base_model
        if 'base_state_dict' in locals():
            del base_state_dict
        if 'clean_state_dict' in locals():
            del clean_state_dict
        if 'residual_model' in locals():
            del residual_model
        if 'reloaded_model_loaded' in locals():
            del reloaded_model_loaded
        
        # Leere den CUDA-Cache, um den Speicher wirklich freizugeben
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("✅ Speicherbereinigung abgeschlossen. Starte Quantisierung...")

        
        # --- CACHING LOGIC END ---

        # Phase 4: Quantize W_res with different bit configurations
        quantization_configs = [
            {"bits": 2, "group_size": 32},
            # {"bits": 2, "group_size": 64},  # Different group size for comparison
            # {"bits": 2, "group_size": 128},
            {"bits": 3, "group_size": 32},
            {"bits": 4, "group_size": 32},
            # {"bits": 4, "group_size": 64},  # Different group size for comparison
            # {"bits": 4, "group_size": 128},
        ]
        quantized_models_info = []

        # --- START OF CUSTOM RE-QUANTIZATION LOGIC ---
        for i, qconfig in enumerate(quantization_configs):
            bits = qconfig["bits"]
            group_size = qconfig["group_size"]

            print(f"\n{'=' * 60}")
            print(f"Custom Re-Quantizing [{i + 1}/{len(quantization_configs)}]: {bits}-bit, group_size={group_size}")
            print(f"{'=' * 60}")

            model_name_clean = script_args.model_name_or_path.replace("/", "_").replace("\\", "_")
            quantized_path = os.path.join(base_output_dir, f"quantized_W_res_{bits}bit_gs{group_size}_{model_name_clean}")

            if os.path.exists(quantized_path) and os.path.exists(os.path.join(quantized_path, "config.json")):
                print(f"⏭️  Quantized model already exists: {quantized_path}")
                quantized_models_info.append({"status": "existing", "bits": bits, "group_size": group_size, "quantized_path": quantized_path})
                continue

            # Lade die Quantisierungsparameter von W_q und die LR-Matrizen
            quantized_path_base_model = os.path.join(base_output_dir, f"quantized_base_model_{model_name_clean}_{bits}bit_gs{group_size}")
            if not os.path.exists(quantized_path_base_model):
                 # Erstelle das quantisierte Basismodell, falls es nicht existiert
                print(f"Quantisiere das Basismodell für {bits}-bit, gs={group_size}...")
                gptq_config_base = GPTQConfig(bits=bits, dataset="c4", tokenizer=tokenizer, group_size=group_size, desc_act=False, sym=False, static_groups=False, true_sequential=True, actorder=True)
                quantized_base_model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path, device_map="auto", quantization_config=gptq_config_base, torch_dtype=torch.float16)
                quantized_base_model.save_pretrained(quantized_path_base_model)
                del quantized_base_model
                torch.cuda.empty_cache()

            state_dict_Wq = load_file(f"{quantized_path_base_model}/model.safetensors")
            LR_matrices = reconstruct_low_rank_update_from_adapter(adapter_path)
            
            clean_state_dict_lr = {}
            for key, value in LR_matrices.items():
                if "base_model.model" in key:
                    clean_key = key.replace("base_model.model.", "")
                    clean_state_dict_lr[clean_key] = value.clone()

            print(f"✅ Loaded {len(state_dict_Wq)} tensors from W_q and {len(clean_state_dict_lr)} LR matrices.")

            final_state_dict = {}
            
            for layer_name, LR_matrix in clean_state_dict_lr.items():
                print(f"  - Processing layer: {layer_name}")
                
                # Hole die originalen quantisierten Parameter
                qweight = state_dict_Wq[f"{layer_name}.qweight"]
                qzeros = state_dict_Wq[f"{layer_name}.qzeros"]
                scales = state_dict_Wq[f"{layer_name}.scales"]
                
                # GPTQModel speichert die Gewichte transponiert. Wir müssen die LR-Matrix anpassen.
                LR_matrix_T = LR_matrix.T.contiguous()

                unpacked_qweight = unpack_qweight(qweight, bits)
                unpacked_qzeros = unpack_qzeros(qzeros, bits)

                scales_expanded = scales.repeat_interleave(group_size, dim=0)
                unpacked_qzeros_expanded = unpacked_qzeros.repeat_interleave(group_size, dim=0)

                assert unpacked_qweight.shape == LR_matrix_T.shape, f"Shape mismatch: W_q.T {unpacked_qweight.shape} vs LR.T {LR_matrix_T.shape}"
                assert scales_expanded.shape == unpacked_qweight.shape, f"Shape mismatch: Scales {scales_expanded.shape} vs W_q.T {unpacked_qweight.shape}"
                assert unpacked_qzeros_expanded.shape == unpacked_qweight.shape, f"Shape mismatch: Zeros {unpacked_qzeros_expanded.shape} vs W_q.T {unpacked_qweight.shape}"

                dtype = scales.dtype
                unpacked_qweight_float = unpacked_qweight.to(dtype)
                unpacked_qzeros_float = unpacked_qzeros_expanded.to(dtype)
                
                reconstructed_W = (unpacked_qweight_float - unpacked_qzeros_float) * scales_expanded

                reconstructed_W_res = reconstructed_W - LR_matrix_T.to(dtype)
                
                # Re-quantisiere mit den *alten* scales und zeros
                # new_q_float = (W_res / scale) + qzero
                new_qweight_float_requant = torch.round(reconstructed_W_res / scales_expanded) + unpacked_qzeros_float

                # 4. Clippe und packe die neuen Gewichte wieder zusammen
                max_val = (1 << bits) - 1
                new_qweight_clipped = torch.clamp(new_qweight_float_requant, 0, max_val).to(torch.int32)
                
                new_qweight_packed = pack_qweight(new_qweight_clipped, bits)
                
                final_state_dict[f"{layer_name}.qweight"] = new_qweight_packed
                final_state_dict[f"{layer_name}.qzeros"] = qzeros # Behalte die originalen gepackten qzeros
                final_state_dict[f"{layer_name}.scales"] = scales # Behalte die originalen scales


            # Füge alle nicht-modifizierten Tensoren hinzu
            for key, value in state_dict_Wq.items():
                if key not in final_state_dict:
                    final_state_dict[key] = value
            
            print("\n✅ Re-quantization complete. Comparing changes...")
            
            # Load the original quantized model to compare against
            original_model = AutoModelForCausalLM.from_pretrained(quantized_path_base_model, device_map="auto")
            original_state_dict = original_model.state_dict()
            
            # Compare qweight changes layer by layer
            total_qweights = 0
            total_changed = 0
            layer_changes = {}
            
            for layer_name in clean_state_dict_lr.keys():
                qweight_key = f"{layer_name}.qweight"
                if qweight_key in original_state_dict and qweight_key in final_state_dict:
                    original_qweight = original_state_dict[qweight_key]
                    new_qweight = final_state_dict[qweight_key]
                    
                    # Ensure tensors are on the same device
                    if original_qweight.device != new_qweight.device:
                        new_qweight = new_qweight.to(original_qweight.device)
                    
                    # Count total elements
                    layer_total = original_qweight.numel()
                    total_qweights += layer_total
                    
                    # Count changed elements
                    changed_mask = (original_qweight != new_qweight)
                    layer_changed = changed_mask.sum().item()
                    total_changed += layer_changed
                    
                    # Calculate percentage for this layer
                    layer_percentage = (layer_changed / layer_total) * 100 if layer_total > 0 else 0
                    
                    layer_changes[layer_name] = {
                        "total": layer_total,
                        "changed": layer_changed,
                        "percentage": layer_percentage
                    }
                    
                    print(f"  - {layer_name}: {layer_changed}/{layer_total} qweights changed ({layer_percentage:.2f}%)")
            
            # Overall statistics
            overall_percentage = (total_changed / total_qweights) * 100 if total_qweights > 0 else 0
            
            print(f"\n📊 QUANTIZATION IMPACT SUMMARY:")
            print(f"  - Total qweights analyzed: {total_qweights:,}")
            print(f"  - Total qweights changed: {total_changed:,}")
            print(f"  - Overall change percentage: {overall_percentage:.2f}%")
            print(f"  - Layers processed: {len(layer_changes)}")
            
            # Show top 5 most changed layers
            if layer_changes:
                sorted_layers = sorted(layer_changes.items(), key=lambda x: x[1]['percentage'], reverse=True)
                print(f"\n🔥 Top 5 most impacted layers:")
                for i, (layer_name, stats) in enumerate(sorted_layers[:5]):
                    print(f"  {i+1}. {layer_name}: {stats['percentage']:.2f}% changed")
            
            # Clean up original model to free memory
            del original_model, original_state_dict
            torch.cuda.empty_cache()
            
            # Now load the final model and apply the new state dict
            print("\n🔄 Loading final model with modified qweights...")
            final_model = AutoModelForCausalLM.from_pretrained(quantized_path_base_model, device_map="auto")
            final_model.load_state_dict(final_state_dict, strict=True)
            
            # Speichere das Endergebnis
            os.makedirs(quantized_path, exist_ok=True)
            final_model.save_pretrained(quantized_path)
            tokenizer.save_pretrained(quantized_path)
            
            # Add the comparison results to the quantized model info
            quantized_models_info.append({
                "status": "created", 
                "bits": bits, 
                "group_size": group_size, 
                "quantized_path": quantized_path,
                "qweight_changes": {
                    "total_qweights": total_qweights,
                    "changed_qweights": total_changed,
                    "change_percentage": overall_percentage,
                    "most_impacted_layer": sorted_layers[0][0] if sorted_layers else None,
                    "max_layer_change_percentage": sorted_layers[0][1]['percentage'] if sorted_layers else 0
                }
            })
            
            # Speicherbereinigung
            del final_model, state_dict_Wq, LR_matrices, clean_state_dict_lr, final_state_dict
            torch.cuda.empty_cache()

            print(f"✅ {bits}-bit quantization complete!")

            # --- old quantization code end ---


        # Phase 5: Save summary of all quantized models
        summary = {
            "base_model": script_args.model_name_or_path,
            "lora_config": {
                "r": script_args.lora_r,
                "alpha": 2 * script_args.lora_r,
                "target_modules": target_modules,
                "initialization": "daniel",
            },
            "adapter_path": adapter_path,
            "quantized_models": quantized_models_info,
            "total_created": len([x for x in quantized_models_info if x["status"] == "created"]),
            "total_existing": len([x for x in quantized_models_info if x["status"] == "exists"]),
            "total_failed": len([x for x in quantized_models_info if x["status"] == "failed"]),
        }

        summary_path = os.path.join(
            base_output_dir, f"quantization_summary_r{script_args.lora_r}_{model_name_clean}.json"
        )
        with open(summary_path, "w") as f:
            import json

            json.dump(summary, f, indent=2)

        # Cleanup temporary files
        import shutil

        # shutil.rmtree(temp_residual_path)

        print(f"\n{'=' * 60}")
        print("🎉 QUANTIZATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Base model: {script_args.model_name_or_path}")
        print(f"LoRA rank: {script_args.lora_r}")
        print(f"Adapter saved to: {adapter_path}")
        print(f"Created: {summary['total_created']} quantized models")
        print(f"Existing: {summary['total_existing']} quantized models")
        print(f"Failed: {summary['total_failed']} quantizations")
        print(f"Summary saved to: {summary_path}")
        print("\nQuantized models:")
        for info in quantized_models_info:
            status_emoji = {"created": "✅", "exists": "⏭️", "failed": "❌"}[info["status"]]
            print(f"  {status_emoji} {info['bits']}-bit (gs={info['group_size']}) -> {info['quantized_path']}")

        # Set model to one of the quantized versions for potential continued training
        if quantized_models_info and quantized_models_info[0]["quantized_path"]:
            print(f"\nLoading 4-bit quantized model for training continuation...")
            four_bit_model = [x for x in quantized_models_info if x["bits"] == 4 and x["group_size"] == 32]
            if four_bit_model:
                model = AutoModelForCausalLM.from_pretrained(
                    four_bit_model[0]["quantized_path"], device_map="auto", torch_dtype=torch.float16
                )
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
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
    # trainer = Trainer(model=model, tokenizer=tokenizer, args=script_args, **data_module)
    # trainer.train()
    # trainer.save_state()
    model.save_pretrained(os.path.join(script_args.output_dir, "ft"))
    tokenizer.save_pretrained(os.path.join(script_args.output_dir, "ft"))


if __name__ == "__main__":
    train()
