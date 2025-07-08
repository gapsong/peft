#!/usr/bin/env python3
"""
Training script for fine-tuning language models with QALoRA using GPTQ quantization.
This script supports cached quantization to avoid repeating expensive quantization processes.
"""

import argparse
import os
from gptqmodel.utils.perplexity import Perplexity

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPTQConfig,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
)


from peft import LoraConfig, get_peft_model, PeftModel
from peft.tuners.lora import GPTQLoraLinear


class RequantizeCallback(TrainerCallback):
    """
    A custom callback to re-quantize the base model's weights by merging the LoRA adapters.
    Handles proper dimension alignment for scales and zeros tensors.
    """

    def __init__(self, ppl_evaluator: Perplexity, requantize_every: int = 5):
        """
        Initialize the RequantizeCallback with a perplexity evaluator.

        Args:
            ppl_evaluator (Perplexity): The perplexity evaluator to use for validation
            requantize_every (int): How often to perform requantization (in steps)
        """
        self.ppl_evaluator = ppl_evaluator
        self.requantize_every = requantize_every
        self.last_requantize_step = 0
        self.adapter_data = {}

    def monitor_adapters(self, model, step):
        """Monitor adapter sizes in a simple function"""
        print(f"\n📊 Step {step} - Adapter Sizes:")

        for name, module in model.named_modules():
            if isinstance(module, GPTQLoraLinear):
                base_weights = module.base_layer.dequantize_weight()
                base_norm = torch.norm(base_weights).item()

                # Get LoRA delta
                for active_adapter in module.active_adapter:
                    lora_A = module.lora_A[active_adapter].weight
                    lora_B = module.lora_B[active_adapter].weight
                    scaling = module.scaling[active_adapter]
                    lora_delta = scaling * (lora_B @ lora_A)
                    lora_norm = torch.norm(lora_delta).item()

                    # Relative size
                    relative_size = lora_norm / base_norm if base_norm > 0 else 0

                    # Store data
                    layer_name = name.split(".")[-1]
                    if layer_name not in self.adapter_data:
                        self.adapter_data[layer_name] = []
                    self.adapter_data[layer_name].append((step, relative_size))

                    print(f"  {layer_name}: LoRA/Base = {relative_size:.6f}")

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PeftModel = None,
        **kwargs,
    ):
        """Event called at the end of a training step."""
        # Only requantize every nth step to avoid overhead
        if state.global_step % self.requantize_every != 0:
            return
        self.monitor_adapters(model, state.global_step)

        print(f"\n--- Step {state.global_step}: Merging adapter and re-quantizing model ---")

        is_training = model.training
        model.eval()

        successful_merges = 0
        total_layers = 0
        perplexity_scores = self.ppl_evaluator.calculate()
        if perplexity_scores:
            print(f"{self.requantize_every} perplexity calculated. Before merge  score: {perplexity_scores[-1]:.4f}")

        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, GPTQLoraLinear):
                    base_quant_layer = module.base_layer
                    # print(base_quant_layer.qweight)
                    base_weights = base_quant_layer.dequantize_weight()

                    for active_adapter in module.active_adapter:
                        if active_adapter in module.lora_A:
                            lora_A = module.lora_A[active_adapter].weight
                            lora_B = module.lora_B[active_adapter].weight
                            scaling = module.scaling[active_adapter]

                            # Calculate LoRA delta: scaling * (B @ A)
                            lora_delta = scaling * (lora_B @ lora_A)
                            # print(f"LoRA delta shape: {lora_delta.shape}")
                            # print(f"LoRA scaling factor: {scaling}")

                            # 3. Get combined weights (base + LoRA)
                            combined_weights = base_weights + lora_delta.T
                            base_quant_layer.quantize_to_int(combined_weights)
                    # 1. Originalgewichte holen
                    # dequant = base_quant_layer.dequantize_weight()
                    # print("dequant1", base_quant_layer.quantize_to_int(combined_weights)[0][:5])
                    # dequant2 = base_quant_layer.dequantize_weight()
                    # print("dequant2", base_quant_layer.quantize_to_int(dequant2)[0][:5])
                    # dequant3 = base_quant_layer.dequantize_weight()
                    # print("dequant3", base_quant_layer.quantize_to_int(dequant3)[0][:5])

        print(f"Successfully merged {successful_merges}/{total_layers} layers")

        perplexity_scores = self.ppl_evaluator.calculate()
        if perplexity_scores:
            print(f"{self.requantize_every} perplexity calculated. Final score: {perplexity_scores[-1]:.4f}")

        # Reset LoRA adapters if any merges were successful
        if successful_merges > 0:
            print("Resetting LoRA adapter weights after merging")
            model.reset_adapter()

        # Restore model's training state
        if is_training:
            model.train()


def load_or_quantize_model(
    base_model: str, tokenizer, bits: int = 4, cache_dir: str = "./quantized_models"
) -> AutoModelForCausalLM:
    """
    Load a pre-quantized model from cache or quantize and cache a new one.
    Automatically detects if the model is already GPTQ-quantized.

    Args:
        base_model: Model identifier or path
        tokenizer: Tokenizer for the model
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
            print(f"Model {base_model} is not GPTQ-quantized. Will quantize it.")
            # Clean up the test model to free memory
            del test_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    except Exception as e:
        print(f"Could not load model {base_model} directly: {e}")
        print("Will attempt to quantize it...")

    # If we get here, the model needs to be quantized
    os.makedirs(cache_dir, exist_ok=True)
    model_id = base_model.replace("/", "_").replace("\\", "_")  # Handle Windows paths too
    quantized_model_path = os.path.join(cache_dir, f"{model_id}_gptq_{bits}bit")

    # Check if we already have a cached quantized version
    if os.path.exists(quantized_model_path) and os.path.exists(os.path.join(quantized_model_path, "config.json")):
        print(f"Loading pre-quantized model from cache: {quantized_model_path}")
        return AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto")

    print(f"Quantizing model and saving to cache: {quantized_model_path}")

    # Configure GPTQ for first-time quantization
    gptq_config = GPTQConfig(
        bits=bits,
        dataset="c4",
        tokenizer=tokenizer,
        group_size=32,
        desc_act=False,
        sym=False,
    )

    # Load and quantize the model
    model = AutoModelForCausalLM.from_pretrained(
        base_model, device_map="auto", quantization_config=gptq_config, torch_dtype=torch.float16
    )

    # Save the quantized model to cache
    print(f"Saving quantized model to {quantized_model_path}")
    model.save_pretrained(quantized_model_path)
    tokenizer.save_pretrained(quantized_model_path)

    return model


def tokenize_and_preprocess(examples, tokenizer, max_length: int = 128):
    """
    Tokenize text data and prepare it for language modeling.

    Args:
        examples: Dataset examples with 'text' field
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length

    Returns:
        Processed examples with input_ids and labels
    """
    # Tokenize the text with truncation and padding
    tokenized_output = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)

    # Preprocess labels (set pad tokens to -100 for loss masking)
    labels = tokenized_output["input_ids"].copy()
    labels = [[-100 if token == tokenizer.pad_token_id else token for token in seq] for seq in labels]
    tokenized_output["labels"] = labels

    return tokenized_output


def verify_saved_model(saved_path, original_model=None):
    """Verify the saved model and check its dimensions"""
    print(f"\n🔍 Verifying saved model: {saved_path}")
    print("-" * 50)

    try:
        # Load the saved model
        loaded_model = AutoModelForCausalLM.from_pretrained(saved_path, device_map="auto")
        print(f"✅ Model loaded successfully from {saved_path}")

        # Check GPTQ layers
        gptq_layers = []
        for name, module in loaded_model.named_modules():
            if hasattr(module, "qzeros"):
                gptq_layers.append(
                    {
                        "name": name,
                        "qzeros_shape": tuple(module.qzeros.shape),
                        "qweight_shape": tuple(module.qweight.shape) if hasattr(module, "qweight") else None,
                        "type": type(module).__name__,
                    }
                )

        print(f"GPTQ layers found: {len(gptq_layers)}")
        for layer in gptq_layers[:3]:  # Show first 3
            print(f"  {layer['name']}: qzeros={layer['qzeros_shape']}, qweight={layer['qweight_shape']}")

        # Compare with original if provided
        if original_model:
            orig_gptq = [
                (name, module.qzeros.shape)
                for name, module in original_model.named_modules()
                if hasattr(module, "qzeros")
            ]
            print(f"\nOriginal had {len(orig_gptq)} GPTQ layers")
            print(f"Saved has {len(gptq_layers)} GPTQ layers")
            print("✅ Match!" if len(orig_gptq) == len(gptq_layers) else "❌ Mismatch!")

        return loaded_model

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def train_model(
    base_model: str,
    data_path: str,
    data_split: str,
    output_dir: str,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    cutoff_len: int,
    use_qalora: bool,
    eval_step: int,
    save_step: int,
    device: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: str,
    push_to_hub: bool,
    qalora_group_size: int,
    bits: int,
) -> None:
    """
    Train a model with QALoRA and GPTQ quantization.

    Args:
        base_model: Base model to fine-tune
        data_path: Dataset path
        output_dir: Directory to save model outputs
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        cutoff_len: Maximum sequence length
        val_set_size: Validation set size
        use_dora: Whether to use DoRA
        use_qalora: Whether to use QALoRA
        quantize: Whether to use quantization
        eval_step: Steps between evaluations
        save_step: Steps between saving checkpoints
        device: Device to use (cuda:0, etc.)
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout rate
        lora_target_modules: Target modules for LoRA
        push_to_hub: Whether to push to Hugging Face Hub
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    hf_token = os.getenv("HF_TOKEN")
    device = torch.device(device)
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct", token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load or quantize model
    model = load_or_quantize_model(base_model, tokenizer, bits=bits)

    # Configure LoRA
    target_modules = (
        lora_target_modules.split(",")
        if lora_target_modules
        else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    print("use_qalora", use_qalora)
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        use_qalora=True,
        qalora_group_size=qalora_group_size,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )

    # Get PEFT model with adapters
    model = get_peft_model(model, lora_config)

    # --- DEBUGGING-SCHRITT HINZUFÜGEN ---
    print("--- Model structure after applying PEFT ---")
    print(model)
    print("-----------------------------------------")
    # ------------------------------------

    # Verify the model was created correctly
    model.print_trainable_parameters()

    print("\nCalculating initial perplexity on the test set...")
    # Instantiate the Perplexity evaluator with parameters from the training arguments
    ppl_evaluator = Perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset_path=data_path,
        dataset_name=data_split if data_split else None,  # Pass None if the split string is empty
        split="test",  # Evaluate on the test split
        text_column="text",  # Standard text column name
    )
    # Calculate perplexity using the sequence length and batch size from training arguments
    perplexity_scores = ppl_evaluator.calculate()
    if perplexity_scores:
        print(f"Initial perplexity calculated. Final score: {perplexity_scores[-1]:.4f}")

    model.print_trainable_parameters()

    # Move model to device if not already there
    if device.type != "cuda" or not hasattr(model, "device") or model.device.type != "cuda":
        model = model.to(device)

    # Load and prepare dataset
    dataset = load_dataset(data_path, data_split)

    tokenized_datasets = {
        "train": dataset["train"].map(
            lambda x: tokenize_and_preprocess(x, tokenizer, max_length=cutoff_len),
            batched=True,
            remove_columns=["text"],
            load_from_cache_file=True,
        ),
        "test": dataset["test"].map(
            lambda x: tokenize_and_preprocess(x, tokenizer, max_length=cutoff_len),
            batched=True,
            remove_columns=["text"],
            load_from_cache_file=True,
        ),
    }

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=eval_step,
        save_steps=save_step,
        save_total_limit=2,
        push_to_hub=push_to_hub,
        gradient_accumulation_steps=16,
        fp16=True,
        learning_rate=learning_rate,
        hub_token=hf_token,
        label_names=["labels"],
    )

    # Clear CUDA cache to free memory
    torch.cuda.empty_cache()

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        callbacks=[RequantizeCallback(ppl_evaluator)],  # Add your custom callback here
    )

    # Start training
    print("\nStarting training...")
    trainer.train()

    # Save the final model
    if push_to_hub:
        trainer.push_to_hub(commit_message="Fine-tuned model with QALoRA")

    # Always save locally
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nTraining complete. Model saved to {output_dir}")

    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LLMs with QALoRA and GPTQ quantization")

    # Model and dataset parameters
    parser.add_argument("--base_model", type=str, default="TheBloke/Llama-2-7b-GPTQ", help="Base model path or name")
    parser.add_argument(
        "--data_path", type=str, default="timdettmers/openassistant-guanaco", help="Dataset path or name"
    )
    parser.add_argument("--data_split", type=str, default="", help="Dataset path or name")

    parser.add_argument(
        "--output_dir", type=str, default="./qalora_output", help="Output directory for the fine-tuned model"
    )
    parser.add_argument("--bits", type=int, default=4, help="Init quantization bits")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--cutoff_len", type=int, default=128, help="Max sequence length")

    # Adapter configuration
    parser.add_argument("--use_qalora", action="store_true", help="Apply QALoRA")
    parser.add_argument("--qalora_group_size", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument(
        "--lora_target_modules", type=str, default=None, help="Comma-separated list of target modules for LoRA"
    )

    # Training process options
    parser.add_argument("--eval_step", type=int, default=100, help="Evaluation step interval")
    parser.add_argument("--save_step", type=int, default=500, help="Save step interval")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")

    # Hugging Face Hub options
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the model to Hugging Face Hub")

    args = parser.parse_args()

    # If use_qalora isn't explicitly set in args but passed to train_model
    if not args.use_qalora:
        args.use_qalora = True  # Default to True as in the original code

    my_model, my_tokenizer = train_model(
        base_model=args.base_model,
        data_path=args.data_path,
        data_split=args.data_split,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        cutoff_len=args.cutoff_len,
        use_qalora=args.use_qalora,
        eval_step=args.eval_step,
        save_step=args.save_step,
        device=args.device,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        push_to_hub=args.push_to_hub,
        qalora_group_size=args.qalora_group_size,
        bits=args.bits,
    )

    merged_model = my_model.merge_and_unload()
    merged_model.save_pretrained("qalora_output_merged_model_via_beta_shift")

    print("\nCalculating initial perplexity on the test set...")
    # Instantiate the Perplexity evaluator with parameters from the training arguments
    ppl_evaluator = Perplexity(
        model=merged_model,
        tokenizer=my_tokenizer,
        dataset_path=args.data_path,
        dataset_name=args.data_split if args.data_split else None,  # Pass None if the split string is empty
        split="test",  # Evaluate on the test split
        text_column="text",  # Standard text column name
    )
    # Calculate perplexity using the sequence length and batch size from training arguments
    perplexity_scores = ppl_evaluator.calculate()
    if perplexity_scores:
        print(f"Initial perplexity calculated. Final score: {perplexity_scores[-1]:.4f}")
