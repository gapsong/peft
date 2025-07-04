#!/usr/bin/env python3
"""
Merge QALoRA adapters directly into GPTQ quantized model's parameters.
This script handles SafeTensors format adapter weights and correctly
adjusts packed qzeros.
"""

import argparse
import json
import os
from typing import Optional

import torch
from safetensors.torch import load_file as load_safetensors
from transformers import AutoModelForCausalLM


def load_adapter_weights(adapter_path: str) -> dict[str, torch.Tensor]:
    """Load adapter weights from saved PEFT adapter folder, supporting SafeTensors"""
    adapter_safetensors = os.path.join(adapter_path, "adapter_model.safetensors")
    adapter_bin = os.path.join(adapter_path, "adapter_model.bin")

    if os.path.exists(adapter_safetensors):
        print(f"Loading adapter weights from SafeTensors: {adapter_safetensors}")
        return load_safetensors(adapter_safetensors, device="cpu")
    elif os.path.exists(adapter_bin):
        print(f"Loading adapter weights from bin file: {adapter_bin}")
        return torch.load(adapter_bin, map_location="cpu")
    else:
        raise FileNotFoundError(
            f"Adapter weights (adapter_model.safetensors or adapter_model.bin) not found at {adapter_path}"
        )


def load_adapter_config(adapter_path: str) -> dict:
    """Load adapter configuration from saved PEFT adapter folder"""
    config_file = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Adapter config (adapter_config.json) not found at {config_file}")

    with open(config_file) as f:
        return json.load(f)


def merge_lora_into_gptq(
    model_path: str,
    adapter_path: str,
    output_path: str,
    custom_scale: Optional[float] = None,
    custom_group_size: Optional[int] = None,
    amplification_factor: float = 4.0,
    remove_group_size_division: bool = False,
) -> None:
    """
    Merge LoRA adapters directly into GPTQ quantization parameters.

    Args:
        model_path: Path to quantized model.
        adapter_path: Path to LoRA adapters.
        output_path: Path to save merged model.
        custom_scale: Optional scale factor to override default.
        custom_group_size: Optional group size to override detected value.
        amplification_factor: Factor to amplify scale to account for quantization effects.
        remove_group_size_division: If True, don't divide by group_size in adjustment calculation.
    """
    print(f"Loading quantized model from {model_path}")
    # Load model on CPU to avoid OOM during modification, can be moved to GPU later
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=None, torch_dtype=torch.float32)

    # Get model config for group size
    with open(os.path.join(model_path, "config.json")) as f:
        model_config = json.load(f)

    # Get group size from config if not provided
    group_size = custom_group_size or model_config.get("quantization_config", {}).get("group_size", 128)

    # Load adapter weights and config
    print(f"Loading adapters from {adapter_path}")
    adapter_weights = load_adapter_weights(adapter_path)
    adapter_config = load_adapter_config(adapter_path)

    # Calculate scale with amplification
    lora_alpha = adapter_config.get("lora_alpha", 16)
    lora_r = adapter_config.get("r", 8)
    default_scale = lora_alpha / lora_r
    scale = (custom_scale or default_scale) * amplification_factor

    print("Merging with parameters:")
    print(f"  - Group size (for merge formula): {group_size}")
    print(f"  - Base scale (lora_alpha/r): {default_scale} (alpha={lora_alpha}, r={lora_r})")
    print(f"  - Amplification factor: {amplification_factor}")
    print(f"  - Final scale: {scale}")
    print(f"  - Remove group_size division: {remove_group_size_division}")

    # Track modifications
    modified_count = 0

    # Find all A matrices for determining which layers need updating
    lora_a_keys = [k for k in adapter_weights.keys() if "lora_A" in k]
    print(f"Found {len(lora_a_keys)} LoRA adapter layers")

    # Process each layer
    for a_key in lora_a_keys:
        base_key = a_key.replace(".lora_A.weight", "")
        if base_key.startswith("base_model.model."):
            layer_name = base_key.replace("base_model.model.", "")
        else:
            # Handle cases where the prefix might be different or absent
            layer_name = base_key

        b_key = a_key.replace("lora_A", "lora_B")

        if b_key not in adapter_weights:
            print(f"Warning: No B matrix found for {a_key}, skipping")
            continue

        a_matrix = adapter_weights[a_key]
        b_matrix = adapter_weights[b_key]

        lora_contribution = (b_matrix @ a_matrix).t()  # Transpose to match [out_features, in_features]

        # Debug: Check adapter magnitude
        print(f"\n=== ADAPTER DEBUG FOR {layer_name} ===")
        print(
            f"LoRA A matrix stats: min={a_matrix.min():.6f}, max={a_matrix.max():.6f}, mean={a_matrix.mean():.6f}, std={a_matrix.std():.6f}"
        )
        print(
            f"LoRA B matrix stats: min={b_matrix.min():.6f}, max={b_matrix.max():.6f}, mean={b_matrix.mean():.6f}, std={b_matrix.std():.6f}"
        )
        print(
            f"LoRA contribution stats: min={lora_contribution.min():.6f}, max={lora_contribution.max():.6f}, mean={lora_contribution.mean():.6f}, std={lora_contribution.std():.6f}"
        )

        # Check if adapter was actually trained
        non_zero_a = (a_matrix != 0).sum().item()
        non_zero_b = (b_matrix != 0).sum().item()
        print(
            f"Non-zero elements: A={non_zero_a}/{a_matrix.numel()} ({non_zero_a / a_matrix.numel() * 100:.2f}%), B={non_zero_b}/{b_matrix.numel()} ({non_zero_b / b_matrix.numel() * 100:.2f}%)"
        )

        if a_matrix.std() < 1e-6 and b_matrix.std() < 1e-6:
            print(f"WARNING: Adapter {layer_name} appears untrained (very low std deviation)")

        found_module_to_modify = False
        for name, module in model.named_modules():
            # Match the layer name derived from adapter keys to the module name
            # This matching might need to be more robust depending on naming conventions
            if layer_name == name and hasattr(module, "qzeros") and hasattr(module, "scales"):
                found_module_to_modify = True
                print(f"\nProcessing layer: {name} (matched with adapter layer: {layer_name})")
                with torch.no_grad():
                    bits = getattr(module, "bits", 4)
                    mask = (2**bits) - 1

                    # rows_per_group is the number of output features that share one scale/zero group
                    # This is typically the GPTQ group_size (e.g., 128)
                    # module.scales.shape[0] is the number of such groups for output features
                    # lora_contribution.shape[0] is total output features
                    if module.scales.shape[0] == 0:
                        print(f"  Warning: module.scales.shape[0] is 0 for layer {name}. Skipping.")
                        continue
                    rows_per_group = lora_contribution.shape[0] // module.scales.shape[0]

                    expanded_scales = torch.repeat_interleave(module.scales, rows_per_group, dim=0)
                    if expanded_scales.shape != lora_contribution.shape:
                        print(f"  Warning: Shape mismatch after expanding scales for layer {name}.")
                        print(f"    lora_contribution shape: {lora_contribution.shape}")
                        print(f"    module.scales shape: {module.scales.shape}")
                        print(f"    expanded_scales shape: {expanded_scales.shape}")
                        # Attempt to align columns if that's the mismatch
                        if (
                            expanded_scales.shape[0] == lora_contribution.shape[0]
                            and expanded_scales.shape[1] != lora_contribution.shape[1]
                        ):
                            if lora_contribution.shape[1] % expanded_scales.shape[1] == 0:
                                factor = lora_contribution.shape[1] // expanded_scales.shape[1]
                                expanded_scales = torch.repeat_interleave(expanded_scales, factor, dim=1)
                                print(
                                    f"    Attempted column alignment for expanded_scales, new shape: {expanded_scales.shape}"
                                )
                            else:
                                print("    Cannot align columns for expanded_scales. Skipping layer.")
                                continue
                        elif expanded_scales.shape[0] != lora_contribution.shape[0]:
                            print("    Row mismatch for expanded_scales. Skipping layer.")
                            continue

                    # Calculate raw adjustment with optional group_size division removal
                    if remove_group_size_division:
                        raw_adjustment = lora_contribution * scale / expanded_scales
                        print("  Using formula: lora_contribution * scale / expanded_scales (no group_size division)")
                    else:
                        raw_adjustment = lora_contribution * scale / group_size / expanded_scales
                        print("  Using formula: lora_contribution * scale / group_size / expanded_scales")

                    print(f"  - Original qzeros shape: {module.qzeros.shape}, dtype: {module.qzeros.dtype}")
                    print(f"  - Scales shape: {module.scales.shape}")
                    print(f"  - LoRA contribution shape: {lora_contribution.shape}")
                    print(f"  - Raw adjustment shape: {raw_adjustment.shape}")

                    # Debug adjustment magnitude
                    print(
                        f"  - Raw adjustment stats: min={raw_adjustment.min():.6f}, max={raw_adjustment.max():.6f}, mean={raw_adjustment.mean():.6f}, std={raw_adjustment.std():.6f}"
                    )
                    significant_adjustments = torch.abs(raw_adjustment) >= 0.5
                    print(
                        f"  - Adjustments >= 0.5: {significant_adjustments.sum().item()} / {raw_adjustment.numel()} ({significant_adjustments.float().mean() * 100:.2f}%)"
                    )
                    very_significant_adjustments = torch.abs(raw_adjustment) >= 1.0
                    print(
                        f"  - Adjustments >= 1.0: {very_significant_adjustments.sum().item()} / {raw_adjustment.numel()} ({very_significant_adjustments.float().mean() * 100:.2f}%)"
                    )

                    # Show scale statistics
                    print(
                        f"  - GPTQ scales stats: min={expanded_scales.min():.6f}, max={expanded_scales.max():.6f}, mean={expanded_scales.mean():.6f}"
                    )

                    original_qzeros_packed = module.qzeros.clone()
                    new_qzeros_packed = torch.zeros_like(original_qzeros_packed)

                    if module.qzeros.dtype == torch.int32:
                        elements_per_packed_val = 32 // bits
                        print(f"  - Detected int32 packed qzeros, {elements_per_packed_val} elements per int32.")
                    elif module.qzeros.dtype == torch.int8:
                        elements_per_packed_val = 8 // bits
                        print(f"  - Detected int8 packed qzeros, {elements_per_packed_val} elements per int8.")
                    else:
                        print(
                            f"  - qzeros dtype {module.qzeros.dtype} not explicitly handled for packing. Assuming direct storage or unhandled packing."
                        )
                        # Fallback: attempt direct modification if shapes allow, otherwise skip
                        if module.qzeros.shape == raw_adjustment.shape:  # Unlikely for typical GPTQ
                            int_adjustment = raw_adjustment.round().to(module.qzeros.dtype)
                            module.qzeros = torch.clamp(module.qzeros + int_adjustment, 0, mask)
                            print(f"    Applied direct adjustment to qzeros of dtype {module.qzeros.dtype}")
                        else:
                            # If qzeros are [out_groups, in_features_unpacked]
                            if (
                                module.qzeros.shape[0] == module.scales.shape[0]
                                and module.qzeros.shape[1] == raw_adjustment.shape[1]
                            ):
                                # Average raw_adjustment over output feature groups
                                reshaped_adj = raw_adjustment.reshape(
                                    module.qzeros.shape[0], rows_per_group, raw_adjustment.shape[1]
                                )
                                avg_adj_over_out_groups = reshaped_adj.mean(dim=1)
                                int_adjustment = avg_adj_over_out_groups.round().to(module.qzeros.dtype)
                                module.qzeros = torch.clamp(module.qzeros + int_adjustment, 0, mask)
                                print(f"    Applied averaged adjustment to qzeros of shape {module.qzeros.shape}")
                            else:
                                print(
                                    f"    Cannot directly adjust qzeros of shape {module.qzeros.shape} with raw_adjustment of shape {raw_adjustment.shape}. Skipping qzeros modification for this layer."
                                )
                        modified_count += 1
                        print(f"Modified (or attempted direct modification for): {name}")
                        break  # Break from inner model.named_modules() loop

                    # Track changes for debugging
                    total_changes = 0
                    significant_changes = 0

                    # Iterate over each 4-bit (or n-bit) segment within the packed qzeros
                    for i in range(elements_per_packed_val):
                        shift = i * bits
                        unpacked_zeros_this_segment = (
                            original_qzeros_packed >> shift
                        ) & mask  # Shape: e.g., [12, 72] for SmolLM

                        # This will store the integer adjustment for each 4-bit zero point in this segment
                        segment_int_adjustment = torch.zeros_like(
                            unpacked_zeros_this_segment, dtype=original_qzeros_packed.dtype
                        )

                        # Iterate over output feature groups (e.g., 0 to 11 for SmolLM q_proj qzeros)
                        for g_out in range(original_qzeros_packed.shape[0]):
                            # Iterate over packed input feature groups (e.g., 0 to 71 for SmolLM q_proj qzeros)
                            for g_in_packed in range(original_qzeros_packed.shape[1]):
                                # Determine the actual input feature column this specific 4-bit zero-point corresponds to.
                                current_input_col_scalar_idx = g_in_packed * elements_per_packed_val + i

                                if current_input_col_scalar_idx >= raw_adjustment.shape[1]:
                                    # This can happen if qzeros packing is denser than raw_adjustment columns
                                    continue

                                # Get the slice of raw_adjustment. This specific 4-bit zero point
                                # (unpacked_zeros_this_segment[g_out, g_in_packed]) applies to:
                                # - All output features in group g_out (rows_per_group of them)
                                # - The specific input feature current_input_col_scalar_idx
                                raw_adj_slice_for_this_zero = raw_adjustment[
                                    g_out * rows_per_group : (g_out + 1) * rows_per_group, current_input_col_scalar_idx
                                ]  # Shape: [rows_per_group] e.g., [128]

                                # Average the float adjustment over the output features in this group
                                mean_float_adj_for_this_zero = raw_adj_slice_for_this_zero.mean()

                                # Convert to integer adjustment (this is the delta for the 4-bit qzero)
                                int_adj_for_this_zero = mean_float_adj_for_this_zero.round().to(
                                    original_qzeros_packed.dtype
                                )
                                segment_int_adjustment[g_out, g_in_packed] = int_adj_for_this_zero

                                # Track changes
                                if int_adj_for_this_zero != 0:
                                    total_changes += 1
                                    if abs(int_adj_for_this_zero) >= 1:
                                        significant_changes += 1

                        # Apply adjustment: z_new = z_orig + adjustment_to_z_quantized
                        adjusted_segment_zeros = unpacked_zeros_this_segment + segment_int_adjustment
                        adjusted_segment_zeros = torch.clamp(
                            adjusted_segment_zeros, 0, mask
                        )  # Clamp to 0-15 for 4-bit

                        # Store this adjusted segment for repacking
                        new_qzeros_packed = (new_qzeros_packed & ~(mask << shift)) | (
                            (adjusted_segment_zeros & mask) << shift
                        )

                    module.qzeros = new_qzeros_packed
                    modified_count += 1
                    print(
                        f"  - Applied {total_changes} total changes, {significant_changes} significant changes (>=1)"
                    )
                    print(f"Modified packed qzeros for: {name}")
                    break  # Break from inner model.named_modules() loop

        if not found_module_to_modify:
            print(f"Warning: No matching quantized module found for adapter layer: {layer_name}")

    print(f"\nSuccessfully modified {modified_count} layers.")

    # Save the modified model
    print(f"Saving merged model to {output_path}")
    os.makedirs(output_path, exist_ok=True)  # Ensure output directory exists
    model.save_pretrained(output_path)

    # If there's a tokenizer with the base model, copy it too
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]
    copied_tokenizer = False
    for tf_name in tokenizer_files:
        src_tf_path = os.path.join(model_path, tf_name)
        if os.path.exists(src_tf_path):
            if not copied_tokenizer:
                print("Copying tokenizer files to output directory...")
                copied_tokenizer = True
            dest_tf_path = os.path.join(output_path, tf_name)
            # Simple file copy
            with open(src_tf_path, "rb") as f_src, open(dest_tf_path, "wb") as f_dst:
                f_dst.write(f_src.read())

    print("Merge completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge QALoRA adapters directly into GPTQ model's qzeros.")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the base quantized GPTQ model directory."
    )
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the saved LoRA adapter directory.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the merged model directory.")
    parser.add_argument("--scale", type=float, help="Custom LoRA scaling factor (default: lora_alpha / r).")
    parser.add_argument(
        "--group_size",
        type=int,
        help="GPTQ quantization group_size used in the merge formula (default: read from model config, typically 128).",
    )
    parser.add_argument(
        "--amplification_factor",
        type=float,
        default=4.0,
        help="Factor to amplify scale to account for quantization effects (default: 4.0).",
    )
    parser.add_argument(
        "--remove_group_size_division",
        action="store_true",
        help="Don't divide by group_size in adjustment calculation (may help with small adjustments).",
    )

    args = parser.parse_args()

    merge_lora_into_gptq(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        custom_scale=args.scale,
        custom_group_size=args.group_size,
        amplification_factor=args.amplification_factor,
        remove_group_size_division=args.remove_group_size_division,
    )
