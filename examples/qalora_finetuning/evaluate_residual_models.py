#!/usr/bin/env python3
"""
Evaluation script for pre-quantized residual connection models
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any
import torch

# Import your existing eval functions
from eval_peft import load_model_and_tokenizer, evaluate_with_lm_eval, print_results


def parse_residual_model_info(model_path: str) -> Dict[str, Any]:
    """Extract rank, bits, and group_size from residual model path"""
    path_name = os.path.basename(model_path)
    
    # Pattern 1: w_res_HuggingFaceTB_SmolLM2-1.7B_r256_daniel_4bit_gs32 (quantized residual)
    pattern = r'w_res_(.+)_r(\d+)_daniel_(\d+)bit_gs(\d+)'
    match = re.match(pattern, path_name)
    
    if match:
        base_model_part, rank, bits, group_size = match.groups()
        return {
            "base_model_name": base_model_part.replace("_", "/"),
            "rank": int(rank),
            "bits": int(bits),
            "group_size": int(group_size),
            "model_type": "residual_quantized"
        }
    
    # Pattern 2: temp_residual_base_r256_fp16 (unquantized residual base)
    fp16_pattern = r'temp_residual_base_r(\d+)_fp16'
    match = re.match(fp16_pattern, path_name)
    
    if match:
        rank = match.group(1)
        return {
            "base_model_name": "HuggingFaceTB/SmolLM2-1.7B",  # Default base model
            "rank": int(rank),
            "bits": 16,  # FP16
            "group_size": None,
            "model_type": "residual_fp16_base"
        }
    
    # Skip standalone adapters - they are only used as components
    adapter_pattern = r'daniel_adapter_r(\d+)_(.+)'
    match = re.match(adapter_pattern, path_name)
    
    if match:
        # Don't return this as a standalone model
        return None
    
    return None


def discover_residual_models(base_dir: str) -> List[Dict[str, Any]]:
    """Discover all residual models in the directory"""
    models = []
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        
        # Skip JSON files
        if item.endswith('.json'):
            continue
            
        if os.path.isdir(item_path):
            # Check if it's a valid model directory
            adapter_config = os.path.join(item_path, "adapter_config.json")
            config_json = os.path.join(item_path, "config.json")
            
            # Check if it's either an adapter or a base model
            if os.path.exists(adapter_config) or os.path.exists(config_json):
                model_info = parse_residual_model_info(item)
                if model_info:
                    model_info["path"] = item_path
                    models.append(model_info)
                    print(f"🔍 Found model: {item} -> Type: {model_info['model_type']}, Rank: {model_info['rank']}, Bits: {model_info['bits']}")
    
    return sorted(models, key=lambda x: (x["rank"], x["bits"], x.get("group_size", 0)))


def find_corresponding_adapter(base_model_path: str, base_dir: str) -> str:
    """Find the corresponding adapter for a quantized base model"""
    base_name = os.path.basename(base_model_path)
    
    # Extract rank from base model name: w_res_HuggingFaceTB_SmolLM2-1.7B_r256_daniel_4bit_gs32
    match = re.search(r'_r(\d+)_', base_name)
    if not match:
        return None
    
    rank = match.group(1)
    
    # Extract model name part: HuggingFaceTB_SmolLM2-1.7B
    model_match = re.search(r'w_res_(.+)_r\d+_daniel', base_name)
    if not model_match:
        return None
    
    model_part = model_match.group(1)
    
    # Look for corresponding adapter: daniel_adapter_r256_HuggingFaceTB_SmolLM2-1.7B
    adapter_name = f"daniel_adapter_r{rank}_{model_part}"
    adapter_path = os.path.join(base_dir, adapter_name)
    
    if os.path.exists(adapter_path):
        return adapter_path
    
    return None

def find_corresponding_adapter_for_fp16(base_model_path: str, base_dir: str) -> str:
    """Find the corresponding adapter for an FP16 base model"""
    base_name = os.path.basename(base_model_path)
    
    # Extract rank from FP16 base model name: temp_residual_base_r256_fp16
    match = re.search(r'temp_residual_base_r(\d+)_fp16', base_name)
    if not match:
        return None
    
    rank = match.group(1)
    
    # For FP16 base models, we need to find the adapter based on the directory structure
    # Look in parent directory for adapters with same rank
    parent_dir = os.path.dirname(base_model_path)
    
    # Pattern: daniel_adapter_r256_*
    for item in os.listdir(parent_dir):
        adapter_pattern = f"daniel_adapter_r{rank}_"
        if item.startswith(adapter_pattern) and os.path.isdir(os.path.join(parent_dir, item)):
            adapter_path = os.path.join(parent_dir, item)
            return adapter_path
    
    return None

def evaluate_residual_model(
    model_info: Dict[str, Any],
    base_dir: str,
    tasks: str,
    num_fewshot: int = 5,
    limit: int = None,
    per_device_eval_batch_size: int = 1
) -> Dict[str, Any]:
    """Evaluate a single residual model"""
    
    model_path = model_info["path"]
    
    print(f"\n🔍 Evaluating: {os.path.basename(model_path)}")
    print(f"   Rank: {model_info['rank']}, Bits: {model_info['bits']}, Group Size: {model_info.get('group_size', 'N/A')}")
    
    try:
        # Determine base model and adapter paths
        if model_info["model_type"] == "residual_quantized":
            # This is a quantized base model, find its adapter
            base_model_path = model_path
            adapter_path = find_corresponding_adapter(model_path, base_dir)
            
            if not adapter_path:
                raise ValueError(f"Could not find corresponding adapter for {os.path.basename(model_path)}")
            
            print(f"   Base model: {os.path.basename(base_model_path)}")
            print(f"   Adapter: {os.path.basename(adapter_path)}")
        elif model_info["model_type"] == "residual_fp16_base":
            # This is an unquantized FP16 base model, find its adapter
            base_model_path = model_path
            adapter_path = find_corresponding_adapter_for_fp16(model_path, base_dir)
            
            if not adapter_path:
                raise ValueError(f"Could not find corresponding adapter for {os.path.basename(model_path)}")
            
            print(f"   Base model (FP16): {os.path.basename(base_model_path)}")
            print(f"   Adapter: {os.path.basename(adapter_path)}") 
        elif model_info["model_type"] == "original_adapter":
            # This is an adapter, we need the original base model (unquantized)
            adapter_path = model_path
            base_model_path = model_info["base_model_name"]  # This should be the HF model name
            
            print(f"   Base model: {base_model_path}")
            print(f"   Adapter: {os.path.basename(adapter_path)}")
        
        else:
            raise ValueError(f"Unknown model type: {model_info['model_type']}")
        
        # Load model
        model, tokenizer = load_model_and_tokenizer(adapter_path, base_model_path)
        
        # Run evaluation
        results = evaluate_with_lm_eval(
            model=model,
            tokenizer=tokenizer,
            tasks=tasks,
            num_fewshot=num_fewshot,
            limit=limit,
            per_device_eval_batch_size=per_device_eval_batch_size
        )
        
        # Clean up
        del model
        torch.cuda.empty_cache()
       
        print_results(results)

        return {
            "model_info": model_info,
            "evaluation_results": results["results"],
            "status": "success"
        }
        
    except Exception as e:
        print(f"❌ Error evaluating {model_path}: {e}")
        return {
            "model_info": model_info,
            "evaluation_results": None,
            "status": "failed",
            "error": str(e)
        }

def create_residual_performance_table(results: List[Dict], output_dir: str):
    """Create LaTeX table matching Overleaf format with WikiText column"""
    
    # Sort and group by rank
    sorted_results = sorted([r for r in results if r["status"] == "success"],
                          key=lambda x: (x["model_info"]["rank"], x["model_info"]["bits"], x["model_info"].get("group_size", 0)))
    
    rank_groups = {}
    for r in sorted_results:
        rank = r["model_info"]["rank"]
        if rank not in rank_groups:
            rank_groups[rank] = []
        rank_groups[rank].append(r)
    
    # Generate LaTeX content
    latex = []
    
    # SmolLM2 1.7B Block with multirow for all ranks
    total_rows = sum(len(models) for models in rank_groups.values())
    latex.append(f"\\multirow{{{total_rows}}}{{*}}{{\\shortstack{{SmolLM2 \\\\ 1.7B}}}}")
    
    sorted_ranks = sorted(rank_groups.keys())
    
    for rank_idx, rank in enumerate(sorted_ranks):
        models = rank_groups[rank]
        
        # Sort models: FP16 first, then by bits and group_size
        models.sort(key=lambda x: (0 if x["model_info"]["bits"] == 16 else 1, 
                                   x["model_info"]["bits"], 
                                   x["model_info"].get("group_size", 0)))
        
        for model_idx, result in enumerate(models):
            info = result["model_info"]
            eval_res = result["evaluation_results"]
            
            # Rank column with multirow
            if model_idx == 0:
                rank_col = f"\\multirow{{{len(models)}}}{{*}}{{{rank}}}"
            else:
                rank_col = ""
            
            # Quantization description
            if info["bits"] == 16:
                quant_desc = "16 (FP16)"
            elif info["bits"] == 4 and info.get("group_size"):
                quant_desc = f"4-bit (Gs {info['group_size']})"
            else:
                quant_desc = f"{info['bits']}-bit"
            
            # Extract metrics
            row_data = ["&", rank_col, "&", quant_desc]
            
            # Standard evaluation tasks
            for task in ["arc_challenge", "arc_easy", "boolq", "hellaswag", "openbookqa", "piqa", "winogrande"]:
                if task in eval_res:
                    tr = eval_res[task]
                    acc = tr.get("acc_norm,none", tr.get("acc,none"))
                    stderr = tr.get("acc_norm_stderr,none", tr.get("acc_stderr,none"))
                    if acc is not None and stderr is not None:
                        row_data.append(f"& {acc*100:.1f} ± {stderr*100:.1f}")
                    else:
                        row_data.append("& ")
                else:
                    row_data.append("& ")
            
            # WikiText perplexity (lower is better, no error bars)
            if "wikitext" in eval_res:
                wt = eval_res["wikitext"]
                perplexity = wt.get("word_perplexity,none")
                if perplexity is not None:
                    row_data.append(f"& {perplexity:.1f}")
                else:
                    row_data.append("& ")
            else:
                row_data.append("& ")
            
            row_data.append("\\\\")
            latex.append(" ".join(row_data))
        
        # Add cmidrule after each rank group (except the last one)
        if rank_idx < len(sorted_ranks) - 1:
            latex.append("\\cmidrule{2-12}")  # Updated for 12 columns
    
    # Add final midrule
    latex.append("\\midrule")
    
    # Save to file
    latex_file = os.path.join(output_dir, "residual_quantization_analysis.tex")
    with open(latex_file, "w") as f:
        f.write("\n".join(latex))
    
    print(f"📄 LaTeX saved to: {latex_file}")
    print(f"\n{'='*60}\n" + "\n".join(latex) + f"\n{'='*60}")
    
def main():
    parser = argparse.ArgumentParser(description="Evaluate pre-quantized residual connection models")
    
    # Input configuration
    parser.add_argument("--residual_models_dir", type=str, 
                       default="/home/nudel/Documents/peft/train_results_debugger/quantized_residuals",
                       help="Directory containing quantized residual models")
    
    # Evaluation configuration
    parser.add_argument("--tasks", type=str, 
                       default="arc_challenge,arc_easy,boolq,hellaswag,openbookqa,piqa,winogrande",
                       help="Evaluation tasks")
    parser.add_argument("--num_fewshot", type=int, default=5, help="Number of few-shot examples")
    parser.add_argument("--limit", type=int, help="Limit number of samples for testing")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Evaluation batch size")
    
    # Filtering options
    parser.add_argument("--ranks", type=str, help="Comma-separated ranks to evaluate (e.g., '256,512')")
    parser.add_argument("--bits", type=str, help="Comma-separated bits to evaluate (e.g., '2,3,4')")
    
    # Output configuration  
    parser.add_argument("--output_dir", type=str, default="./residual_eval_results", 
                       help="Output directory for results")
    parser.add_argument("--save_results", action="store_true", help="Save results to JSON")
    parser.add_argument("--generate_latex", action="store_true", help="Generate LaTeX table")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Discover models
    print(f"🔍 Discovering models in: {args.residual_models_dir}")
    all_models = discover_residual_models(args.residual_models_dir)
    
    # Filter models if specified
    if args.ranks:
        target_ranks = [int(r.strip()) for r in args.ranks.split(",")]
        all_models = [m for m in all_models if m["rank"] in target_ranks]
    
    if args.bits:
        target_bits = [int(b.strip()) for b in args.bits.split(",")]
        all_models = [m for m in all_models if m["bits"] in target_bits]
    
    print(f"📊 Found {len(all_models)} models to evaluate:")
    for model in all_models:
        print(f"   - {os.path.basename(model['path'])}: r={model['rank']}, {model['bits']}-bit, gs={model.get('group_size', 'N/A')}")
    
    if not all_models:
        print("❌ No models found to evaluate")
        return
    
    # Run evaluations
    print(f"\n🚀 Starting evaluation of {len(all_models)} models...")
    all_results = []
    
    for i, model_info in enumerate(all_models, 1):
        print(f"\n{'='*60}")
        print(f"🏁 EVALUATING MODEL {i}/{len(all_models)}")
        print(f"{'='*60}")
        
        result = evaluate_residual_model(
            model_info=model_info,
            base_dir=args.residual_models_dir,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            per_device_eval_batch_size=args.per_device_eval_batch_size
        )
        
        all_results.append(result)
        
        if result["status"] == "success":
            print(f"✅ Successfully evaluated {os.path.basename(model_info['path'])}")
        else:
            print(f"❌ Failed to evaluate {os.path.basename(model_info['path'])}")
    
    # Prepare summary
    successful_results = [r for r in all_results if r["status"] == "success"]
    failed_results = [r for r in all_results if r["status"] == "failed"]
    
    print(f"\n📊 EVALUATION SUMMARY:")
    print(f"   ✅ Successful: {len(successful_results)}")
    print(f"   ❌ Failed: {len(failed_results)}")
    
    # Save results
    if args.save_results and successful_results:
        results_file = os.path.join(args.output_dir, "residual_evaluation_results.json")
        
        summary_data = {
            "summary": {
                "total_models": len(all_models),
                "successful_evaluations": len(successful_results),
                "failed_evaluations": len(failed_results),
                "tasks_evaluated": args.tasks,
                "num_fewshot": args.num_fewshot
            },
            "results": all_results
        }
        
        with open(results_file, "w") as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {results_file}")
    
    # Generate LaTeX table
    if args.generate_latex and successful_results:
        print("\n📄 Generating LaTeX table...")
        create_residual_performance_table(successful_results, args.output_dir)
    
    print("\n✅ Residual model evaluation completed!")


if __name__ == "__main__":
    main()