#!/usr/bin/env python3
"""
Evaluation script for PEFT models using lm-eval-harness.
"""

import argparse
import json
import os

import datasets
import lm_eval
import torch
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from gptqmodel import GPTQModel
from peft import PeftModel


datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True


def load_model_and_tokenizer(model_name_or_path, base_model=None):
    """Load model and tokenizer from path"""

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check if this is a PEFT model
    adapter_config_path = os.path.join(model_name_or_path, "adapter_config.json")

    if os.path.exists(adapter_config_path):
        print(f"Loading PEFT model from {model_name_or_path}")

        # Read adapter config
        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)

        # Get base model name
        # hier muss das basemodel mit dem residual model austausgetauscht werden
        base_model_name = adapter_config.get("base_model_name_or_path")
        if base_model:
            base_model_name = base_model

        if not base_model_name:
            raise ValueError("Base model not specified and not found in adapter config")

        print(f"Base model: {base_model_name}")
        # base_model_name = "/home/nudel/Documents/peft/train_results_debugger/quantized_residuals/temp_residual_base"
        # quantized version here
        # base_model_name = "/home/nudel/Documents/peft/train_results_debugger/quantized_residuals/w_res_HuggingFaceTB_SmolLM2-1.7B_r256_daniel_4bit_gs32"
        
        # base_model_name = "/home/nudel/Documents/peft/train_results_debugger/quantized_residuals/w_res_HuggingFaceTB_SmolLM2-1.7B_r256_daniel_4bit_gs128"
        
        # base_model_name = "/home/nudel/Documents/peft/train_results_debugger/quantized_residuals/w_res_HuggingFaceTB_SmolLM2-1.7B_r256_daniel_3bit_gs32"
        # Load base model
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model_name, device_map="auto", torch_dtype=torch.float32
        )
        # model_name_or_path = "/home/nudel/Documents/peft/train_results_debugger/quantized_residuals/daniel_adapter_r256_HuggingFaceTB_SmolLM2-1.7B"
        # Load PEFT model
        model = PeftModel.from_pretrained(base_model_obj, model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
        # model = base_model_obj
        model.eval()

        print(f"✅ PEFT model loaded successfully")

    else:
        print(f"Loading regular model from {model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")

    return model, tokenizer


def test_model_generation(model, tokenizer, test_prompts=None):
    """Test model with simple prompts"""

    if test_prompts is None:
        test_prompts = [
            "Question: What is 15 + 7?\nAnswer:",
            "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat is 25 - 8?\n\n### Response:",
            "<s>[INST] What is 12 * 3? [/INST]",
        ]

    print("\n🧪 Testing model generation:")

    for i, prompt in enumerate(test_prompts):
        print(f"\n--- Test {i + 1} ---")
        print(f"Prompt: {prompt}")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        clean_response = response.split("\n")[0].strip()
        print(f"Response: '{clean_response}'")


def evaluate_with_lm_eval(model, tokenizer, tasks, num_fewshot=5, limit=None, per_device_eval_batch_size=1):
    """Evaluate model using lm-eval-harness"""

    # Create lm-harness model
    lm_harness_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=per_device_eval_batch_size,
        device="cuda",
        trust_remote_code=True,
    )

    # Parse tasks
    if isinstance(tasks, str):
        task_list = [t.strip() for t in tasks.split(",")]
    else:
        task_list = tasks

    print(f"🏃 Running evaluation on tasks: {task_list}")
    print(f"Few-shot examples: {num_fewshot}")
    print(f"Limit: {limit}")

    # Run evaluation
    results = lm_eval.simple_evaluate(
        model=lm_harness_model,
        tasks=task_list,
        num_fewshot=num_fewshot,
        limit=limit,
        batch_size=per_device_eval_batch_size,
    )

    # Clean up
    del lm_harness_model
    torch.cuda.empty_cache()

    return results


def print_results(results):
    """Print evaluation results in a nice format"""

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for task_name, task_results in results["results"].items():
        print(f"\n📊 {task_name.upper()}:")

        for metric_name, value in task_results.items():
            if isinstance(value, (int, float)):
                print(f"  {metric_name}: {value:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate PEFT model using lm-eval-harness")

    # Model parameters (matching corda_finetuning.py naming)
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--base_model", type=str, help="Base model path (if different from training)")

    # Evaluation parameters
    parser.add_argument("--tasks", type=str, default="gsm8k", help="Comma-separated list of tasks to evaluate")
    parser.add_argument("--num_fewshot", type=int, default=5, help="Number of few-shot examples")
    parser.add_argument("--limit", type=int, help="Limit number of samples (for testing)")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Batch size for evaluation")

    # Output parameters
    parser.add_argument("--output_dir", type=str, help="Directory to save results")
    parser.add_argument("--test_generation", action="store_true", help="Test model generation before evaluation")
    parser.add_argument("--apply_gptq_post_quant", action="store_true", help="Quantize model with GPTQ after loading")
    parser.add_argument("--bits", type=int, default=4, help="Limit number of samples (for testing)")

    args = parser.parse_args()

    print(f"🔍 Evaluating model: {args.model_name_or_path}")
    print(f"📝 Tasks: {args.tasks}")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path, args.base_model)

    if args.apply_gptq_post_quant:
        print("Applying GPTQ post-quantization...")
        gptq_config = GPTQConfig(
            bits=args.bits,
            dataset="c4",
            tokenizer=tokenizer,
            group_size=32,
            desc_act=False,
            sym=False,
        )
        model = model.merge_and_unload()
        temp_model_path = "./temp_merged_model"

        model.save_pretrained(temp_model_path)

        # post quantization gptq
        model = AutoModelForCausalLM.from_pretrained(
            temp_model_path,
            quantization_config=gptq_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    # Run evaluation
    results = evaluate_with_lm_eval(
        model=model,
        tokenizer=tokenizer,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
    )

    # Print results
    print_results(results)

    # Save results if requested
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, "eval_results.json")

        # Remove samples to reduce file size
        results_clean = results.copy()
        if "samples" in results_clean:
            del results_clean["samples"]

        with open(output_path, "w") as f:
            json.dump(results_clean, f, indent=2, default=str)

        print(f"📁 Results saved to: {output_path}")

    # Clean up
    del model
    torch.cuda.empty_cache()

    print("✅ Evaluation complete!")


if __name__ == "__main__":
    main()
