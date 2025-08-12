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

def evaluate_residual_model(
    model_info: Dict[str, Any],
    tasks: str,
    num_fewshot: int = 5,
    limit: int = None,
    per_device_eval_batch_size: int = 1
) -> Dict[str, Any]:
    """Evaluate a single residual model"""
    
    model_path = model_info["path"]
    base_model = model_info["base_model_name"]
    
    print(f"\n🔍 Evaluating: {os.path.basename(model_path)}")
    print(f"   Rank: {model_info['rank']}, Bits: {model_info['bits']}, Group Size: {model_info.get('group_size', 'N/A')}")
    
    # try:
    # Load model
    base_model = "/home/nudel/Documents/peft/train_results_debugger/quantized_residuals_r256/w_res_HuggingFaceTB_SmolLM2-1.7B_r256_daniel_4bit_gs128"
    adapter_path = "/home/nudel/Documents/peft/train_results_debugger/quantized_residuals_r256/daniel_adapter_r256_HuggingFaceTB_SmolLM2-1.7B"
    model, tokenizer = load_model_and_tokenizer(adapter_path, base_model)
    
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
        
    # except Exception as e:
    #     print(f"❌ Error evaluating {model_path}: {e}")
    #     return {
    #         "model_info": model_info,
    #         "evaluation_results": None,
    #         "status": "failed",
    #         "error": str(e)
    #     }


def create_residual_performance_table(results: List[Dict], output_dir: str):
    """Create LaTeX table specifically for residual connection analysis"""
    
    # Group results by rank
    rank_groups = {}
    for result in results:
        if result["status"] != "success":
            continue
            
        rank = result["model_info"]["rank"]
        if rank not in rank_groups:
            rank_groups[rank] = []
        rank_groups[rank].append(result)
    
    # Sort each rank group by bits and group_size
    for rank in rank_groups:
        rank_groups[rank].sort(key=lambda x: (
            x["model_info"]["bits"], 
            x["model_info"].get("group_size", 0)
        ))
    
    # Extract task columns
    task_mapping = {
        "arc_challenge": "ARC-c",
        "arc_easy": "ARC-e", 
        "boolq": "BoolQ",
        "hellaswag": "HellaSwag",
        "openbookqa": "OpenBookQA",
        "piqa": "PIQA",
        "winogrande": "Winogrande",
        "gsm8k": "GSM8K",
        "mmlu": "MMLU"
    }
    
    # Get task columns from first successful result
    task_columns = []
    sample_result = next((r for r in results if r["status"] == "success"), None)
    if sample_result:
        for task_name in sample_result["evaluation_results"].keys():
            latex_name = task_mapping.get(task_name, task_name.upper())
            task_columns.append((task_name, latex_name))
    
    # Generate LaTeX
    latex_content = []
    latex_content.extend([
        "\\begin{table}[htbp]",
        "\\tiny",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\caption{Performance evaluation of Daniel residual quantization across different ranks ($r$) and quantization levels. Shows the trade-off between adapter rank, residual compression, and task performance.}",
        "\\label{tab:daniel_residual_quantization_analysis}",
        "\\centering"
    ])
    
    # Table header
    num_cols = 3 + len(task_columns)  # Rank, Bits, Group Size + tasks
    col_spec = "c" * num_cols
    latex_content.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_content.append("\\toprule")
    
    header = ["\\textbf{Rank ($r$)}", "\\textbf{Bits}", "\\textbf{Group Size}"]
    header.extend([f"\\textbf{{{name}}}" for _, name in task_columns])
    latex_content.append(" & ".join(header) + " \\\\")
    latex_content.append("\\midrule")
    
    # Data rows grouped by rank
    for rank in sorted(rank_groups.keys()):
        rank_results = rank_groups[rank]
        num_rank_rows = len(rank_results)
        
        latex_content.append(f"% Rank {rank} block")
        latex_content.append(f"\\multirow{{{num_rank_rows}}}{{*}}{{{rank}}}")
        
        for i, result in enumerate(rank_results):
            model_info = result["model_info"]
            eval_results = result["evaluation_results"]
            
            row_data = [""]  # Rank handled by multirow
            
            # Bits
            if model_info["bits"] == 16:
                row_data.append("FP16")
            else:
                row_data.append(f"{model_info['bits']}")
            
            # Group size
            group_size = model_info.get("group_size")
            row_data.append(str(group_size) if group_size else "--")
            
            # Task results
            for task_name, _ in task_columns:
                if task_name in eval_results:
                    task_result = eval_results[task_name]
                    # Find the main metric for this task
                    main_metric = None
                    for metric_name, value in task_result.items():
                        if isinstance(value, (int, float)) and any(
                            x in metric_name.lower() for x in ["acc", "exact_match", "em", "norm"]
                        ):
                            main_metric = value
                            break
                    
                    if main_metric is not None:
                        row_data.append(f"{main_metric:.3f}")
                    else:
                        row_data.append("--")
                else:
                    row_data.append("--")
            
            latex_content.append(" & ".join(row_data) + " \\\\")
        
        # Add separator between rank groups
        if rank != max(rank_groups.keys()):
            latex_content.append("\\midrule")
    
    latex_content.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    # Save to file
    latex_file = os.path.join(output_dir, "residual_quantization_analysis.tex")
    with open(latex_file, "w") as f:
        f.write("\n".join(latex_content))
    
    print(f"📄 LaTeX table saved to: {latex_file}")
    print(f"\n{'='*80}")
    print("📋 RESIDUAL QUANTIZATION ANALYSIS TABLE:")
    print(f"{'='*80}")
    print("\n".join(latex_content))
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate pre-quantized residual connection models")
    
    # Input configuration
    parser.add_argument("--residual_models_dir", type=str, 
                       default="/home/nudel/Documents/peft/train_results_debugger/quantized_residuals",
                       help="Directory containing quantized residual models")
    
    # Evaluation configuration
    parser.add_argument("--tasks", type=str, 
                       default="arc_challenge,hellaswag,piqa,winogrande,gsm8k",
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