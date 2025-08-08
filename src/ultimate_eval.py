import logging
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPTQConfig,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training


# --- Configuration ---
# Model and Paths
model_id = "meta-llama/Llama-3.2-1B-Instruct"
output_dir_base = "./smollm_eval_results"

# LoRA / QALoRA Parameters
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# GPTQ Parameters
gptq_bits = 4
gptq_group_size = 128
gptq_calibration_samples = 128
gptq_calibration_max_len = 512

# Training Parameters
training_max_seq_length = 512
training_epochs = 1
training_batch_size = 12
training_learning_rate = 2e-4

# --- Setup ---
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Ensure output directory exists
os.makedirs(output_dir_base, exist_ok=True)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
logger.info(f"Using device: {device}, with dtype: {torch_dtype}")


def load_model(model_path, torch_dtype=None, use_gptq=False):
    """Load a model with appropriate configuration."""
    logger.info(f"Loading model from {model_path}")

    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch_dtype,
    }

    # Check if the model is already GPTQ quantized
    model_is_gptq = "gptq" in model_path.lower()
    gptq_save_path = f"{output_dir_base}/model_gptq_{gptq_bits}bit"

    # Add GPTQ specific arguments only if requested AND the model isn't already GPTQ
    if use_gptq and not model_is_gptq:
        logger.info(f"Applying GPTQ quantization config (bits={gptq_bits}, group_size={gptq_group_size})")
        model_kwargs["quantization_config"] = GPTQConfig(
            bits=gptq_bits,
            dataset="c4",
            tokenizer=tokenizer,
            group_size=128,
            desc_act=False,
            sym=False,
        )

        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

        # Only save if we actually quantized
        logger.info(f"Saving quantized GPTQ model to {gptq_save_path}")
        model.save_pretrained(gptq_save_path)
    else:
        # Regular model loading
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    # Resize token embeddings if needed
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))

    return model


def compute_perplexity(model, tokenizer, text, model_name=None):
    """
    Compute perplexity of a model on raw text data
    """
    # Set model to evaluation mode
    model.eval()

    # Join text elements if it's a list
    if isinstance(text, list):
        text = " ".join(text)

    # Tokenize input text
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings["input_ids"].to(model.device)

    # Define max sequence length and stride for processing long texts
    max_length = min(tokenizer.model_max_length, 512)
    stride = 256

    total_log_likelihood = 0
    total_tokens = 0

    # Process text in chunks with overlap
    for i in range(0, input_ids.size(1), stride):
        begin_loc = max(0, i)
        end_loc = min(begin_loc + max_length, input_ids.size(1))

        # Get input chunk
        input_chunk = input_ids[:, begin_loc:end_loc]
        target_chunk = input_chunk.clone()

        # Skip if chunk is too small
        if input_chunk.size(1) < 2:
            continue

        with torch.no_grad():
            outputs = model(input_chunk)
            logits = outputs.logits[:, :-1]  # Remove last position

            # Get target tokens (shifted by 1)
            target = target_chunk[:, 1:]

            # Calculate log probabilities
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_log_probs = torch.gather(log_probs, dim=2, index=target.unsqueeze(-1)).squeeze(-1)

            # Count tokens
            total_log_likelihood += token_log_probs.sum().item()
            total_tokens += target.numel()

    avg_neg_log_likelihood = -total_log_likelihood / total_tokens
    perplexity = torch.exp(torch.tensor(avg_neg_log_likelihood)).item()

    if model_name:
        logger.info(f"Perplexity for {model_name}: {perplexity:.2f}")

    return perplexity


def setup_qalora_model(base_model):
    """Setup a model for QALoRA fine-tuning."""
    # Prepare model for k-bit training
    base_model = prepare_model_for_kbit_training(base_model)

    # Define LoRA configuration
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        inference_mode=False,
        use_qalora=True,
    )

    # Apply LoRA adapters
    lora_model = get_peft_model(base_model, peft_config)

    return lora_model


def get_or_create_tokenized_datasets(
    tokenizer,
    cache_dir="./cached_datasets",
):
    """
    Load tokenized datasets from cache or create and cache them if not found.

    Args:
        tokenizer: The tokenizer to use if datasets need to be created
        cache_dir: Directory to store cached datasets

    Returns:
        dict: Dictionary with "train" and "test" tokenized datasets
    """
    os.makedirs(cache_dir, exist_ok=True)

    train_cache_path = os.path.join(cache_dir, "wikitext_train_tokenized.arrow")

    # Check if cached datasets exist
    # if os.path.exists(train_cache_path):
    #     logger.info("Loading tokenized datasets from cache...")
    #     train_dataset = Dataset.load_from_disk(train_cache_path)
    #     return {"train": train_dataset}

    # If not cached, create and save them
    logger.info("Tokenized datasets not found in cache. Creating new ones...")

    def tokenize_and_preprocess(examples, tokenizer, max_length=512):
        """
        Tokenize text data and prepare it for language modeling.

        Args:
            examples: Dataset examples with 'text' field
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length

        Returns:
            Processed examples with input_ids and labels as PyTorch tensors
        """
        # Tokenize only the first 1000 texts
        texts = examples["text"][:1000]
        tokenized = tokenizer(texts, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")

        # Set up labels for causal LM (same as input_ids)
        labels = tokenized["input_ids"].clone()
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        tokenized["labels"] = labels

        return tokenized

    # Load and prepare dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    tokenized_datasets = {
        "train": dataset["train"]
        .select(range(1000))
        .map(
            lambda x: tokenize_and_preprocess(x, tokenizer, max_length=training_max_seq_length),
            batched=True,
            remove_columns=["text"],
        ),
        "test": dataset["test"].map(
            lambda x: tokenize_and_preprocess(x, tokenizer, max_length=training_max_seq_length),
            batched=True,
            remove_columns=["text"],
        ),
    }
    # Save to cache
    logger.info(f"Saving tokenized datasets to {cache_dir}...")
    tokenized_datasets["train"].save_to_disk(train_cache_path)

    return tokenized_datasets


def train_qalora_model(model, tokenizer):
    """Train a model using QALoRA."""
    # Create data collator
    tokenized_datasets = get_or_create_tokenized_datasets(tokenizer, cache_dir=f"{output_dir_base}/cached_datasets")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"{output_dir_base}/qalora_trained",
        num_train_epochs=training_epochs,
        per_device_train_batch_size=training_batch_size,
        learning_rate=training_learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir=f"{output_dir_base}/logs",
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],  # Properly tokenized with labels
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
    )

    # Train model
    trainer.train()

    # Save model
    trainer.save_model()

    return f"{output_dir_base}/qalora_trained"


# --- Main Evaluation Script ---

if __name__ == "__main__":
    results = {}

    # Load tokenizer once and configure pad token
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load evaluation dataset
    logger.info("Loading wikitext test dataset for perplexity evaluation...")
    wikitext_test_raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]
    # Filter out empty strings which can cause issues
    wikitext_test_raw = [text for text in wikitext_test_raw if text.strip()]

    # --- Experiment 1: Base Full-Precision Model ---
    try:
        logger.info("\n--- Experiment 1: Base Full-Precision Model ---")
        base_model = load_model(model_id, torch_dtype=torch_dtype)
        ppl_base = compute_perplexity(base_model, tokenizer, wikitext_test_raw, "Base Model")
        results["exp1_base_model_ppl"] = ppl_base
        # Free up memory
        del base_model
        torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"Error in Experiment 1: {e}")
        results["exp1_base_model_ppl"] = "Error"

    # --- Experiment 2: Base GPTQ Model ---
    try:
        logger.info("\n--- Experiment 2: Base GPTQ-Quantized Model ---")
        gptq_model_path = f"{output_dir_base}/model_gptq_{gptq_bits}bit"

        # Check if GPTQ model already exists
        if not os.path.exists(os.path.join(gptq_model_path, "config.json")):
            logger.info(f"GPTQ model not found at {gptq_model_path}. Using original GPTQ model.")
            gptq_model = load_model(model_id, torch_dtype=torch_dtype, use_gptq=True)
        else:
            logger.info(f"Loading GPTQ model from {gptq_model_path}")
            gptq_model = load_model(gptq_model_path, torch_dtype=torch_dtype)

        ppl_gptq = compute_perplexity(gptq_model, tokenizer, wikitext_test_raw, "Base GPTQ Model")
        results["exp2_base_gptq_ppl"] = ppl_gptq
        # Keep GPTQ model for next experiment
    except Exception as e:
        logger.error(f"Error in Experiment 2: {e}")
        results["exp2_base_gptq_ppl"] = "Error"
        # Try to load model again if it failed
        try:
            gptq_model = load_model(model_id, torch_dtype=torch_dtype, use_gptq=True)
        except Exception as e:
            logger.error(f"Failed to load GPTQ model: {e}")
            exit(1)

    # --- Experiment 3: QALoRA fine-tuned model ---
    try:
        logger.info("\n--- Experiment 3: QALoRA Fine-tuned Model ---")
        qalora_output_path = f"{output_dir_base}/qalora_trained"

        # Check if QALoRA model exists
        if not os.path.exists(os.path.join(qalora_output_path, "adapter_config.json")):
            logger.info(f"Loading existing QALoRA model from {qalora_output_path}")
            qalora_model = load_model(qalora_output_path, torch_dtype=torch_dtype)
        else:
            logger.info("Setting up and training QALoRA model")
            # Setup QALoRA model using the GPTQ model from previous step
            qalora_model = setup_qalora_model(gptq_model)

            # Load a small training dataset
            logger.info("Loading training data for QALoRA")

            # Train QALoRA model
            qalora_output_path = train_qalora_model(qalora_model, tokenizer)
            ppl_qalora = compute_perplexity(qalora_model, tokenizer, wikitext_test_raw, "QALoRA Fine-tuned Model")

            # Reload the trained model
            qalora_model = load_model(qalora_output_path, torch_dtype=torch_dtype)

        # Evaluate QALoRA model
        ppl_qalora = compute_perplexity(qalora_model, tokenizer, wikitext_test_raw, "QALoRA Fine-tuned Model")
        results["exp3_qalora_model_ppl"] = ppl_qalora
        del qalora_model
        torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"Error in Experiment 3: {e}")
        results["exp3_qalora_model_ppl"] = "Error"

    # Print summary of results
    logger.info("\n--- Summary of Results ---")
    for name, value in results.items():
        logger.info(f"{name}: {value}")
