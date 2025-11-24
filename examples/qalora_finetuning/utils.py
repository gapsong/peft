"""Memory requirement utility

The memory requirement in the tables of this presentation were calculated by
using the included `utils.py` script. To run the script, make sure that
`accelerate` is installed in your Python environment (`python -m pip install
accelerate`). Executing the script does _not_ download the model or load it into
memory. Therefore, you can all this for very large models without the risk to
run out of memory.

```bash
# return memory estimate of Llama3 8B
python utils.py "meta-llama/Meta-Llama-3-8B"
# the same, but using rank 32 for LoRA
python utils.py "meta-llama/Meta-Llama-3-8B" --rank 32
# the same, but loading the model with 4bit quantization
python utils.py "meta-llama/Meta-Llama-3-8B" --dtype int4
```

Note that for gated models, you need to have a Hugging Face account, accept the
terms of the model, and log in to your Hugging Face account:
https://huggingface.co/docs/huggingface_hub/en/quick-start#login-command.

The size of the activations is not included in the estimate.

"""

import argparse
import json
import sys
import warnings
from collections import defaultdict
import os
import torch
from safetensors import safe_open
import transformers
from accelerate.utils.other import convert_bytes
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import argparse
import re
import struct
from pathlib import Path

# suppress all warnings and logs
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

# ==========================================
# KONSTANTEN
# ==========================================
LORA = "lora"
QALORA = "qalora"

# Byte-Größen Definitionen
dtype_to_bytes_linear = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "int4": 0.5}
dtype_to_bytes_other = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 2, "int4": 2}

# ==========================================
# HELPER: OFFLINE PARSING (Low Level)
# ==========================================
def read_safetensors_header_direct(file_path: Path):
    """
    Liest den Header einer Safetensors Datei direkt binär.
    Benötigt KEINE externen Libraries (nur Python 'struct').
    """
    try:
        with open(file_path, 'rb') as f:
            # Die ersten 8 Bytes sind ein uint64 (Länge des Headers)
            header_length_bytes = f.read(8)
            if not header_length_bytes: return {}
            
            # Little-Endian Unsigned Long Long (<Q)
            header_length = struct.unpack('<Q', header_length_bytes)[0]

            # Den Header JSON String lesen
            header_bytes = f.read(header_length)
            header = json.loads(header_bytes.decode('utf-8'))
            return header
    except Exception as e:
        print(f"⚠️ Fehler beim Lesen von {file_path.name}: {e}", file=sys.stderr)
        return {}

def convert_bytes(size):
    """Einfache Konvertierung ohne 'accelerate' Library"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"

# ==========================================
# CORE LOGIC
# ==========================================
def analyze_local_model(model_path_str, rank, group_size):
    """
    Analysiert lokale Safetensors Dateien in einem Ordner.
    """
    base_path = Path(model_path_str)
    
    if not base_path.exists():
        print(f"❌ Pfad nicht gefunden: {base_path}", file=sys.stderr)
        return None

    # 1. Dateien finden (Sharding Support)
    files_to_scan = []
    
    if base_path.is_file():
        # User hat direkt auf eine Datei gezeigt
        files_to_scan = [base_path.name]
        base_path = base_path.parent
    elif base_path.is_dir():
        # Check auf index.json (Sharding Index)
        index_file = base_path / "model.safetensors.index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    data = json.load(f)
                    # Deduplizierte Liste aller Split-Dateien
                    files_to_scan = list(set(data["weight_map"].values()))
                # print(f"📂 Sharded Modell erkannt ({len(files_to_scan)} Teile).", file=sys.stderr)
            except Exception as e:
                print(f"⚠️ Index defekt, suche manuell nach .safetensors...", file=sys.stderr)
                files_to_scan = [f.name for f in base_path.glob("*.safetensors")]
        else:
            # Keine Index Datei, wir nehmen alle safetensors im Root
            files_to_scan = [f.name for f in base_path.glob("*.safetensors")]
            if not files_to_scan:
                # Fallback: Manche Repos nennen es 'adapter_model.safetensors'
                files_to_scan = [f.name for f in base_path.glob("*model*.safetensors")]

    if not files_to_scan:
        print("❌ Keine .safetensors Dateien im angegebenen Pfad gefunden.", file=sys.stderr)
        return None

    # Regex für Target Modules (Linear Layers)
    target_pattern = re.compile(r".*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|fc[12]|dense)\.(weight|qweight)$")
    
    stats = defaultdict(int)

    # 2. Iteration über Dateien
    for filename in files_to_scan:
        full_path = base_path / filename
        if not full_path.exists():
            continue
            
        header = read_safetensors_header_direct(full_path)
        
        for param_name, info in header.items():
            if param_name == '__metadata__': continue
            
            shape = info.get('shape', [])
            
            # Base Params zählen
            num_params = 1
            for s in shape: num_params *= s
            
            # A) Overhead (Scales/Zeros)
            if any(x in param_name for x in ["scales", "qzeros", "g_idx", "absmax"]):
                stats["Metadata/Overhead"] += num_params
                continue

            # B) Base Params Classification
            if "weight" in param_name or "qweight" in param_name:
                 stats["Linear.weight"] += num_params
            else:
                 stats["Metadata/Overhead"] += num_params

            # C) LoRA / QALoRA Berechnung
            if target_pattern.match(param_name):
                
                # Dimensionen wiederherstellen
                if "qweight" in param_name:
                    # GPTQ: Shape ist [In/8, Out] -> Wir brauchen [In, Out]
                    # Annahme: 4-bit Quantisierung -> Packing Factor 8
                    # Heuristik: input ist meist die gepackte Dimension
                    dim1 = shape[0]
                    dim2 = shape[1]
                    n_in = dim1 * 8
                    m_out = dim2
                else:
                    # Normal: [Out, In] 
                    m_out = shape[0]
                    n_in = shape[1]

                # LoRA:  (In * r) + (Out * r)
                stats[LORA] += (n_in * rank) + (m_out * rank)
                
                # QALoRA: (In * r / G) + (Out * r)
                stats[QALORA] += ((n_in // group_size) * rank) + (m_out * rank)

    return dict(stats)

# ==========================================
# MEMORY CALCULATION
# ==========================================
def get_param_bytes(count_params, dtype):
    num_bytes = defaultdict(int)
    
    for key, val in count_params.items():
        if key == "Linear.weight":
            num_bytes[key] = int(val * dtype_to_bytes_linear[dtype])
        
        elif key == LORA:
            num_bytes[key] = int(val * dtype_to_bytes_other["float32"])
        
        elif key == QALORA:
            num_bytes[key] = int(val * dtype_to_bytes_other["bfloat16"])
        
        else:
            # Metadata/Overhead
            num_bytes[key] = int(val * dtype_to_bytes_other[dtype])
            
    return dict(num_bytes)


def get_training_memory_estimate(num_bytes):
    adapter_keys = [LORA, QALORA]
    total_size_base = sum(v for k, v in num_bytes.items() if k not in adapter_keys)
    
    # Factor: Model Weights + Gradients + Optimizer States
    # Simplified estimate
    factor = 3 

    estimates = {
        "memory full fine-tuning": total_size_base + (factor * total_size_base),
    }

    if LORA in num_bytes:
        estimates["memory LoRA fine-tuning"] = total_size_base + (factor * num_bytes[LORA])

    if QALORA in num_bytes:
        estimates["memory QALoRA fine-tuning"] = total_size_base + (factor * num_bytes[QALORA])

    return estimates

# ==========================================
# MAIN
# ==========================================
def main(model_path, rank, group_size, dtype, sink=print):
    
    # 1. Struktur Analyse (Lokal)
    count_params = analyze_local_model(model_path, rank=rank, group_size=group_size)
    
    if not count_params:
        return

    # 2. Byte Berechnung
    num_bytes = get_param_bytes(count_params, dtype=dtype)
    num_bytes_readable = {k: convert_bytes(v) for k, v in num_bytes.items()}

    adapter_keys = [LORA, QALORA]

    # Stats aggregieren
    total_params_base = sum(v for k, v in count_params.items() if k not in adapter_keys)
    total_size_base = sum(v for k, v in num_bytes.items() if k not in adapter_keys)
    
    total_params_with_adapters = sum(count_params.values())
    total_size_with_adapters = sum(num_bytes.values())

    training_bytes = get_training_memory_estimate(num_bytes)
    training_bytes_readable = {k: convert_bytes(v) for k, v in training_bytes.items()}

    # JSON Result
    result = {
        "number of parameters": count_params,
        "number of bytes": num_bytes,
        "number of bytes (readable)": num_bytes_readable,
        "total number of parameters (base model)": total_params_base,
        "total number of parameters (with adapters)": total_params_with_adapters,
        "total size (base model)": total_size_base,
        "total size (with adapters)": total_size_with_adapters,
        "total size (base model, readable)": convert_bytes(total_size_base),
        "total size (with adapters, readable)": convert_bytes(total_size_with_adapters),
    }

    # Estimates einfügen
    if "memory full fine-tuning" in training_bytes_readable:
        result["memory required for full fine-tuning"] = training_bytes_readable["memory full fine-tuning"]
    if "memory LoRA fine-tuning" in training_bytes_readable:
        result["memory required for LoRA fine-tuning"] = training_bytes_readable["memory LoRA fine-tuning"]
    if "memory QALoRA fine-tuning" in training_bytes_readable:
        result["memory required for QALoRA fine-tuning"] = training_bytes_readable["memory QALoRA fine-tuning"]

    if dtype.startswith("int"):
        if "memory required for full fine-tuning" in result:
            result["memory required for full fine-tuning"] += "*"

    sink(json.dumps(result, indent=2))
    
    if dtype.startswith("int"):
        print("*Note that quantized models cannot be fine-tuned without PEFT", file=sys.stderr)
    
    return result

def calc_qalora_stats(model_path, rank, group_size):
    """Berechnet LoRA vs QALoRA Params offline (Direct Header Read)."""
    path = Path(model_path)
    if not path.exists(): return None

    # 1. Dateien finden (Sharding Support)
    files = []
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, 'r') as f: files = list(set(json.load(f)["weight_map"].values()))
    else:
        files = [f.name for f in path.glob("*.safetensors")]

    # 2. Scannen
    stats = {"lora_params": 0, "qalora_params": 0}
    # Regex für Linear Layers (Llama, Mistral, etc.)
    layer_re = re.compile(r".*\.(q|k|v|o|gate|up|down|fc\d|dense)\.(weight|qweight)$")

    for fname in files:
        try:
            with open(path / fname, 'rb') as f:
                # Header Länge lesen (uint64) + Header parsen
                header_len = struct.unpack('<Q', f.read(8))[0]
                header = json.loads(f.read(header_len).decode('utf-8'))
            
            for name, info in header.items():
                if not layer_re.match(name): continue
                
                shape = info['shape']
                # GPTQ (qweight) vs Normal Handling
                if "qweight" in name:
                    n_in, m_out = shape[0] * 8, shape[1] # Annahme: 4-bit Packing
                else:
                    m_out, n_in = shape[0], shape[1] # Standard: [Out, In]

                stats["lora_params"] += (n_in * rank) + (m_out * rank)
                stats["qalora_params"] += ((n_in // group_size) * rank) + (m_out * rank)
        except: continue

    stats["saved_params"] = stats["lora_params"] - stats["qalora_params"]
    return stats

def load_all_lr_layers(adapter_path):
    """
    Lädt einen gespeicherten LoRA-Adapter, extrahiert lora_A und lora_B,
    berechnet den Skalierungsfaktor und rekonstruiert die Low-Rank-Update-Matrix für alle Layer.
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
    # Verwende safe_open für safetensors
    adapter_state_dict = {}
    lr_layers = {}
    with safe_open(weights_path, framework="pt") as f:
        for key in f.keys():
            adapter_state_dict[key] = f.get_tensor(key)
    print(f"✅ {len(adapter_state_dict)} Tensoren geladen.\n")

    # 5. Durch alle Layer iterieren und die Low-Rank-Matrix rekonstruieren
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
            
            # Speichere im Dict (base_key ist der Layer-Name, z.B. "q_proj")
            cleaned_keys = base_key + ".base_layer.weight"
            lr_layers[cleaned_keys] = low_rank_update
            
            print(f"--- Rekonstruiert für Layer: {base_key} ---")
            print(f"  - lora_A Form: {lora_A.shape}")
            print(f"  - lora_B Form: {lora_B.shape}")
            print(f"  - Rekonstruierte LR-Matrix Form: {low_rank_update.shape}\n")

    print(f"✅ Alle LR-Layer geladen und in lr_layers gespeichert: {list(lr_layers.keys())}")
    
    return lr_layers

def apply_lr_weights(model, lr_layers):
    """
    Wendet die Low-Rank-Gewichte auf das Modell an.
    """
    for name, param in model.named_parameters():
        if name in lr_layers:
            print(f"🔄 Wende LR-Gewichte auf Layer '{name}' an.")
            param.data = lr_layers[name].to(param.device, dtype=param.dtype)

def compare_weights(model, lr_layers):
    """
    Vergleicht die ursprünglichen Gewichte des Modells mit den aktualisierten Gewichten.
    Gibt die maximale, minimale und durchschnittliche Abweichung aus.
    """
    total_error = 0.0
    total_params = 0

    for name, param in model.named_parameters():
        if name in lr_layers:
            original_weights = param.data.clone()
            updated_weights = original_weights + lr_layers[name].to(param.device, dtype=param.dtype)
            error = torch.abs(updated_weights - original_weights).sum().item()
            total_error += error
            total_params += param.numel()

            print(f"🔍 Layer '{name}':")
            print(f"  - Max Error: {torch.abs(updated_weights - original_weights).max().item():.6f}")
            print(f"  - Min Error: {torch.abs(updated_weights - original_weights).min().item():.6f}")
            print(f"  - Mean Error: {error / param.numel():.6f}\n")

    print(f"✅ Gesamtfehler: {total_error:.6f}")
    print(f"✅ Durchschnittlicher Fehler pro Parameter: {total_error / total_params:.6f}")

def show_gradients_of_model(model):
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"❌ Gradient disabled for parameter: {name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", type=str, help="Model name (on Hugging Face)")
    parser.add_argument("--rank", type=int, default=8, help="Rank of LoRA adapter")
    parser.add_argument(
        "--dtype", type=str, default="float32", help="Data type, one of float32, float16, bfloat16, int8, int4"
    )
    parser.add_argument("--group_size", type=int, default=8, help="QA_Lora_groupsize")
    args = parser.parse_args()
    main(args.model_id, rank=args.rank, dtype=args.dtype, group_size=args.group_size)


# =================================================================================
# 1. DATA LOADING UTILITY (Angepasst von Ihrem Code)
# =================================================================================
def get_c4_calibration_data(tokenizer, n_samples=128, seq_len=512):
    """
    Lädt das C4-Dataset von Hugging Face und bereitet tokenisierte Samples für die Kalibrierung vor.
    """
    print(f" Lade C4 Kalibrierungsdatensatz (bis zu {n_samples} Samples)...")
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    
    samples = []
    dataset_iterator = iter(dataset)
    
    with tqdm(total=n_samples, desc="Lade Daten") as pbar:
        while len(samples) < n_samples:
            try:
                row = next(dataset_iterator)
                text = row['text']
                # Tokenizer direkt auf dem Gerät anwenden, um Zeit zu sparen, wenn device bekannt ist.
                tokens = tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True)
                if tokens.input_ids.shape[1] == seq_len:
                    samples.append(tokens)
                    pbar.update(1)
            except StopIteration:
                print("Warnung: Ende des Datasets erreicht, bevor n_samples gefüllt wurden.")
                break
    print(f"✅ {len(samples)} Kalibrierungssamples geladen.")
    return samples

# =================================================================================
# 2. CORE FUNCTION: Outlier Calculation using Hessian Diagonal Approximation
# =================================================================================
def calculate_hessian_outliers(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    outlier_percentage: float = 0.001,
    n_samples: int = 128,
    seq_len: int = 512,
    target_modules_substrings: list = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
) -> dict:
    # ... (Docstring and initial setup)
    print(f"🔬 Starting Saliency Calculation (GPTQ method) for Top {outlier_percentage*100:.3f}% Outliers...")
    device = next(model.parameters()).device
    model.eval()

    calibration_data = get_c4_calibration_data(tokenizer, n_samples=n_samples, seq_len=seq_len)

    hessian_diag_approximations = {} 
    sample_counts = defaultdict(int)
    hook_handles = []
    target_layer_modules = {}

    def create_hook(name):
        def capture_inputs_hook(module, args, output):
            x = args[0]
            # CORRECTED LINE HERE
            x_flat = x.view(-1, x.shape[-1]) 
            
            hessian_diag_batch = torch.sum(x_flat.float()**2, dim=0)
            
            if name not in hessian_diag_approximations:
                hessian_diag_approximations[name] = hessian_diag_batch
            else:
                hessian_diag_approximations[name] += hessian_diag_batch
                
            sample_counts[name] += x_flat.shape[0]
        return capture_inputs_hook

    print("   -> Registering forward hooks for target modules...")
    for name, module in model.named_modules():
        if any(substring in name for substring in target_modules_substrings) and isinstance(module, nn.Linear):
            target_layer_modules[name] = module
            handle = module.register_forward_hook(create_hook(name))
            hook_handles.append(handle)

    print(f"   -> Running {len(calibration_data)} calibration samples through the model...")
    with torch.no_grad():
        for batch in tqdm(calibration_data, desc="Calibration Forward Pass"):
            model(batch.input_ids.to(device))

    for handle in hook_handles:
        handle.remove()

    # ... (The rest of the function remains unchanged and is correct)
    all_saliencies = []
    layer_param_info = [] 
    
    print("   -> Calculating saliency scores for all target layers...")
    for name, module in target_layer_modules.items():
        if sample_counts[name] == 0:
            print(f"Warning: Layer {name} received no activations.")
            continue
            
        H_diag = hessian_diag_approximations[name] / sample_counts[name]
        W = module.weight.data
        saliency_matrix = (W.float()**2) * H_diag.unsqueeze(0) / 2.0
        
        all_saliencies.append(saliency_matrix.view(-1))
        layer_param_info.append({"name": name, "num_params": saliency_matrix.numel()})

    all_scores_tensor = torch.cat(all_saliencies)
    
    num_total_params = all_scores_tensor.numel()
    num_outliers_to_keep = int(num_total_params * outlier_percentage)
    
    print(f"   -> Finding global threshold for top {num_outliers_to_keep} of {num_total_params} parameters...")
    if num_outliers_to_keep > 0:
        threshold = torch.kthvalue(all_scores_tensor, num_total_params - num_outliers_to_keep).values
    else:
        threshold = torch.finfo(torch.float32).max

    del all_saliencies, all_scores_tensor
    torch.cuda.empty_cache()

    outlier_results = {}
    print("   -> Extracting outliers based on global threshold...")
    
    total_extracted_count = 0
    for name, module in target_layer_modules.items():
        if sample_counts[name] == 0:
            continue
        
        H_diag = hessian_diag_approximations[name] / sample_counts[name]
        W = module.weight.data
        saliency_matrix = (W.float()**2) * H_diag.unsqueeze(0) / 2.0
        outlier_mask = saliency_matrix >= threshold
        outlier_indices_flat = outlier_mask.view(-1).nonzero().squeeze(-1)

        if outlier_indices_flat.numel() > 0:
            outlier_weights = module.weight.data.view(-1)[outlier_indices_flat].clone().to(torch.bfloat16)
            outlier_results[name] = (outlier_weights, outlier_indices_flat)
            total_extracted_count += outlier_indices_flat.numel()

    print(f"✅ Extraction complete. Extracted a total of {total_extracted_count} outliers from {len(outlier_results)} layers.")
    return outlier_results

    
    
import torch
import torch.nn as nn
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

@torch.no_grad()
def quantize_uniform_symmetric_grouped(W: torch.Tensor, bits: int = 3, group_size: int = 16, axis: int = 1) -> torch.Tensor:
    assert 2 <= bits <= 8
    qmax = (1 << (bits - 1)) - 1
    qmin = -qmax

    W_perm = W.transpose(axis, -1).contiguous()
    orig_shape = W_perm.shape
    C = orig_shape[-1]
    G = (C + group_size - 1) // group_size
    pad = G * group_size - C
    if pad > 0:
        W_pad = torch.nn.functional.pad(W_perm, (0, pad))
    else:
        W_pad = W_perm

    Wg = W_pad.view(-1, G, group_size)
    max_abs = Wg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = max_abs / qmax
    Q = torch.clamp(torch.round(Wg / scale), qmin, qmax)
    Wg_q = Q * scale

    W_pad_q = Wg_q.view(*orig_shape[:-1], G * group_size)
    if pad > 0:
        W_q = W_pad_q[..., :C]
    else:
        W_q = W_pad_q
    return W_q.transpose(-1, axis).contiguous()

@torch.no_grad()
def spqr_extract_outliers(
    model: nn.Module,
    calibration_batches: List[Dict[str, torch.Tensor]],
    target_module_name_substrings: Optional[List[str]] = None,
    outlier_threshold: float = 0.2,
    bits: int = 3,
    group_size: int = 16,
    percdamp: float = 1.0,  # small damping for H_diag
    device: Optional[torch.device] = None,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Returns: dict[layer_name] = (outlier_values_bf16, outlier_flat_indices)
    Threshold is applied to squared Hessian-weighted error:
        (W - Q(W))^2 * H_diag > outlier_threshold
    which mirrors SpQR's normalized error thresholding up to a constant.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    # 1) Register hooks to accumulate sum of squares of inputs per Linear
    h_sums: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = defaultdict(int)
    linears: Dict[str, nn.Linear] = {}

    def make_hook(name: str):
        def hook(module: nn.Module, inputs, output):
            x = inputs[0]  # inputs is a tuple
            x = x.to(dtype=torch.float32)
            x_flat = x.reshape(-1, x.shape[-1])  # [..., C_in] -> [N, C_in]
            s = (x_flat * x_flat).sum(dim=0)  # [C_in]
            if name not in h_sums:
                h_sums[name] = s.clone()
            else:
                h_sums[name] += s
            counts[name] += x_flat.shape[0]
        return hook

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if target_module_name_substrings is not None:
                if not any(sub in name for sub in target_module_name_substrings):
                    continue
            linears[name] = module
            handles.append(module.register_forward_hook(make_hook(name)))

    # 2) Run calibration forward to collect activations
    with torch.no_grad():
        for batch in calibration_batches:
            # Expect HF-style batch dict with "input_ids" at minimum
            inputs = {k: v.to(device) for k, v in batch.items()}
            model(**inputs) if "input_ids" in inputs else model(inputs)

    for h in handles:
        h.remove()

    # 3) Compute outliers using grouped quantization and H_diag
    results: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for name, lin in linears.items():
        if counts[name] == 0:
            continue
        W = lin.weight.data  # [C_out, C_in]
        H_diag = (h_sums[name] / counts[name]).to(W.device).to(torch.float32)  # [C_in]
        H_diag = H_diag + percdamp  # damping for stability

        Wq = quantize_uniform_symmetric_grouped(W, bits=bits, group_size=group_size, axis=1)
        diff2 = (W - Wq).to(torch.float32).pow(2)  # [C_out, C_in]
        # Hessian-weighted squared error per weight (proportional to SpQR's criterion)
        weighted_err = diff2 * H_diag.unsqueeze(0)  # [C_out, C_in]

        mask = weighted_err > float(outlier_threshold)
        idx = mask.view(-1).nonzero(as_tuple=False).squeeze(-1)  # flat indices
        if idx.numel() > 0:
            vals = W.view(-1)[idx].detach().clone().to(torch.bfloat16)
            results[name] = (vals, idx)

    return results

@torch.no_grad()
def build_calibration_batches(
    tokenizer,
    texts: List[str],
    n_samples: int = 128,
    seq_len: int = 512,
) -> List[Dict[str, torch.Tensor]]:
    """
    Tokenize texts into a list of small dicts: {'input_ids': ..., 'attention_mask': ...}
    Batches are single-sequence to keep memory small; concatenate outside if you want larger batches.
    """
    batches: List[Dict[str, torch.Tensor]] = []
    for t in texts[:n_samples]:
        enc = tokenizer(
            t,
            truncation=True,
            max_length=seq_len,
            return_tensors="pt",
            padding=False,
        )
        batches.append({k: v.squeeze(0) for k, v in enc.items()})  # squeeze batch dim for HF CausalLM
    return batches