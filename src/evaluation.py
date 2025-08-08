import enum
import math
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import get_task_dict


class EvalMethod(str, enum.Enum):
    """Evaluation method options"""

    HARNESS = "harness"  # LM Evaluation Harness (external benchmarks)
    CUSTOM = "custom"  # Token-by-token accuracy and perplexity
    SLIDING = "sliding_window"  # Sliding window perplexity
    BATCHED = "batched"  # Batched perplexity calculation


@dataclass
class EvalConfig:
    """Configuration for model evaluation"""

    method: EvalMethod = EvalMethod.CUSTOM
    task_name: str = "wikitext"  # For harness evaluation
    max_batches: Optional[int] = None
    device: Union[str, int] = "cuda:0"
    verbose: bool = False
    max_length: int = 512
    truncation: bool = True
    dtype: torch.dtype = torch.bfloat16


def evaluate_model(
    model, tokenizer, config: Union[EvalConfig, dict[str, Any], str] = EvalConfig(), testloader=None, dataset=None
) -> tuple[Optional[float], float, float]:
    """
    Unified evaluation function that supports multiple evaluation methods.

    Args:
        model: The language model to evaluate
        tokenizer: Associated tokenizer
        config: Evaluation configuration (EvalConfig, dict, or method name string)
        testloader: DataLoader for custom evaluation methods
        dataset: Raw dataset for some evaluation methods

    Returns:
        Tuple of (accuracy, perplexity, average_loss)

    Examples:
        # Using the default configuration (custom evaluation)
        results = evaluate_model(model, tokenizer, testloader=test_loader)

        # Using a specific method by name
        results = evaluate_model(model, tokenizer, "harness")

        # Using a complete configuration
        config = EvalConfig(
            method=EvalMethod.SLIDING,
            max_batches=10,
            verbose=True
        )
        results = evaluate_model(model, tokenizer, config, testloader=test_loader)
    """
    # Handle different config input types
    if isinstance(config, str):
        config = EvalConfig(method=config)
    elif isinstance(config, dict):
        # Convert dict to EvalConfig
        config = EvalConfig(**config)

    # Verify model type
    if isinstance(model, torch.nn.Module):
        model_class = model.__class__.__name__
    else:
        model_class = model.module.__class__.__name__

    supported_models = ["LlamaForCausalLM", "GPTNeoForCausalLM"]
    if model_class not in supported_models:
        supported = ", ".join(supported_models)
        raise ValueError(f"Unsupported model type: {model_class}. Supported: {supported}")

    # Choose evaluation method
    if config.method == EvalMethod.HARNESS:
        return _evaluate_with_harness(
            model,
            tokenizer=tokenizer,
            task_name=config.task_name,
            device=config.device,
            verbose=config.verbose,
            max_length=config.max_length,
            truncation=config.truncation,
            dtype=config.dtype,
        )
    elif config.method == EvalMethod.CUSTOM:
        if testloader is None:
            raise ValueError("testloader is required for custom perplexity evaluation")
        return _calculate_custom_perplexity(
            model,
            testloader=testloader,
            tokenizer=tokenizer,
            max_batches=config.max_batches,
            device=config.device,
            verbose=config.verbose,
        )
    elif config.method == EvalMethod.SLIDING:
        if testloader is None:
            raise ValueError("testloader is required for sliding window evaluation")
        return _calculate_sliding_window_perplexity(
            model,
            dataloader=testloader,
            device=config.device,
            verbose=config.verbose,
        )
    elif config.method == EvalMethod.BATCHED:
        from perplexity import batched_perplexity

        if dataset is None:
            raise ValueError("dataset is required for batched perplexity evaluation")
        return batched_perplexity(model=model, tokenizer=tokenizer, dataset=dataset)
    else:
        raise ValueError(f"Unknown evaluation method: {config.method}")


def _evaluate_with_harness(
    model,
    tokenizer,
    task_name="wikitext",
    device="cuda:0",
    verbose=False,
    max_length=512,
    truncation=True,
    dtype=torch.bfloat16,
):
    """Evaluate model using the lm-evaluation-harness"""
    # Initialize the harness-compatible model
    hf_model = HFLM(
        pretrained=model,
        backend="causal",
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        trust_remote_code=True,
        max_length=max_length,
        truncation=truncation,
    )

    # Load the task and run evaluation
    task_dict = get_task_dict([task_name])
    results = evaluator.evaluate(
        lm=hf_model, task_dict=task_dict, log_samples=False, verbosity="WARNING" if not verbose else "INFO"
    )

    # Extract metrics
    perplexity = results["results"][task_name]["word_perplexity,none"]
    average_loss = perplexity  # Same as perplexity in this case
    accuracy = None  # Wikitext doesn't provide accuracy

    if verbose:
        print("\nHarness Evaluation Results:")
        print(f" - Word Perplexity: {perplexity:.4f}")
        print(f" - Byte Perplexity: {results['results'][task_name]['byte_perplexity,none']:.4f}")

    return accuracy, perplexity, average_loss


def _calculate_custom_perplexity(model, testloader, tokenizer, max_batches=None, device="cuda:0", verbose=False):
    """Calculate token-level perplexity and accuracy"""
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    correct_predictions = 0

    total_iterations = max_batches or len(testloader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(testloader, start=1):
            if max_batches and batch_idx > max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["input_ids"].clone().to(device)
            pad_token_id = tokenizer.pad_token_id
            labels[labels == pad_token_id] = -100  # Mask padding tokens

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            # Compute accuracy
            predictions = torch.argmax(logits, dim=-1)
            mask = labels != -100
            token_count = mask.sum().item()

            # Weight the loss by number of tokens in this batch
            total_loss += loss.item() * token_count
            correct_predictions += torch.sum((predictions == labels) & mask).item()
            total_tokens += token_count

            if verbose and batch_idx % 50 == 0:
                acc = 100.0 * correct_predictions / total_tokens
                print(f"Batch {batch_idx}/{total_iterations}: Accuracy = {acc:.4f}%")

    # Calculate metrics
    accuracy = 100.0 * correct_predictions / total_tokens
    average_loss = total_loss / total_tokens
    perplexity = 2 ** (average_loss / math.log(2))

    if verbose:
        print("\nCustom Perplexity Results:")
        print(f" - Token Accuracy: {accuracy:.4f}%")
        print(f" - Average Loss: {average_loss:.4f}")
        print(f" - Perplexity: {perplexity:.4f}")

    return accuracy, perplexity, average_loss


def _calculate_sliding_window_perplexity(model, dataloader, device="cuda:0", verbose=False):
    """Calculate perplexity using a sliding window approach"""
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    correct_predictions = 0
    sample_count = len(dataloader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if verbose and batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx}/{sample_count}")

            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()

            # Forward pass with labels
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            logits = outputs.logits

            # Compute accuracy
            predictions = torch.argmax(logits, dim=-1)
            mask = labels != -100
            token_count = mask.sum().item()

            if token_count > 0:
                total_loss += loss.item() * token_count
                correct_predictions += torch.sum((predictions == labels) & mask).item()
                total_tokens += token_count

    # Calculate final metrics
    if total_tokens == 0:
        return 0.0, float("inf"), float("inf")

    accuracy = 100.0 * correct_predictions / total_tokens
    average_loss = total_loss / total_tokens
    perplexity = 2 ** (average_loss / math.log(2))

    if verbose:
        print("\nSliding Window Perplexity Results:")
        print(f" - Processed {sample_count} batches, {total_tokens} tokens")
        print(f" - Token Accuracy: {accuracy:.4f}%")
        print(f" - Average Loss: {average_loss:.4f}")
        print(f" - Perplexity: {perplexity:.4f}")

    return accuracy, perplexity, average_loss
