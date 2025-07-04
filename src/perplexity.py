# Directly taken from https://huggingface.co/spaces/evaluate-measurement/perplexity/blob/main/perplexity.py
# TODO replace with a strided version https://github.com/huggingface/transformers/issues/9648#issuecomment-812981524

import torch
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def nll_loss_no_mean(logits, labels):
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L1228
    logits = logits.float()
    # Shift so that tokens < n predict n
    vocab_size = logits.shape[-1]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=-100, reduce=False)
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    return loss_fct(shift_logits, shift_labels)


def create_batch(input_ids, loss_mask, batch_i, batch_size, stride):
    text_len = input_ids.size(1)

    # create batch inds
    begin_locs, end_locs, trg_lens = [], [], []
    for j in range(batch_size):
        j = batch_i + j * stride
        if j >= text_len:
            break
        begin_loc = max(j, 0)
        end_loc = min(j + stride, text_len)
        trg_len = end_loc - j  # may be different from stride on last loop

        begin_locs.append(begin_loc)
        end_locs.append(end_loc)
        trg_lens.append(trg_len)

    if not begin_locs:  # Handle empty batch case
        return None, None

    # Find max length for padding
    max_len = max(e - b for b, e in zip(begin_locs, end_locs))

    # Create batch with padding
    b_input_ids = []
    b_loss_mask = []

    for b, e in zip(begin_locs, end_locs):
        # Extract segment
        segment_ids = input_ids[:, b:e]
        segment_mask = loss_mask[:, b:e]

        # Calculate padding needed
        pad_len = max_len - segment_ids.size(1)

        # Apply padding if needed
        if pad_len > 0:
            # Use pad token ID from tokenizer or 0 as default
            pad_value = 0  # Change this to tokenizer.pad_token_id if available
            padding = torch.ones(1, pad_len, dtype=segment_ids.dtype, device=segment_ids.device) * pad_value
            segment_ids = torch.cat([segment_ids, padding], dim=1)

            # Pad mask with False (don't compute loss on padding)
            mask_padding = torch.zeros(1, pad_len, dtype=segment_mask.dtype, device=segment_mask.device)
            segment_mask = torch.cat([segment_mask, mask_padding], dim=1)

        b_input_ids.append(segment_ids)
        b_loss_mask.append(segment_mask)

    # Stack into batch
    b_input_ids = torch.stack(b_input_ids, dim=1).squeeze(0)
    b_loss_mask = torch.stack(b_loss_mask, dim=1).squeeze(0)

    # Create target
    target_ids = torch.ones_like(b_input_ids) * -100  # -100 is the default ignore_index value in CrossEntropyLoss

    # Set target IDs for non-padded regions
    for i, (trg_len, total_len) in enumerate(zip(trg_lens, [e - b for b, e in zip(begin_locs, end_locs)])):
        # Only set target for the actual window content, not padding
        labels = b_input_ids[i, :total_len].clone()
        target_start = max(0, total_len - trg_len)
        target_ids[i, target_start:total_len] = labels[target_start:total_len]

    # Apply loss mask
    target_ids[~b_loss_mask] = -100

    return b_input_ids, target_ids


@torch.no_grad()
def batched_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    data_path: str = "Salesforce/wikitext",
    data_split: str = "wikitext-2-raw-v1",
    batch_size: int = 4,  # Kleinere Batch-Größe für die Perplexity-Berechnung
):
    """
    Calculates perplexity in a memory-efficient, batched manner.
    """
    print(f"Calculating perplexity on {data_path}/{data_split}...")
    dataset = load_dataset(data_path, data_split, split="test")
    texts = [text for text in dataset["text"] if text and not text.isspace()]

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i in tqdm(range(0, len(texts), batch_size), desc="Perplexity"):
        batch_texts = texts[i : i + batch_size]

        # Tokenize each batch separately
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
        ).to(model.device)

        labels = inputs.input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding in loss

        outputs = model(**inputs, labels=labels)

        # Accumulate loss and number of tokens
        loss = outputs.loss.item()
        num_tokens = (labels != -100).sum().item()

        total_loss += loss * num_tokens
        total_tokens += num_tokens

    if total_tokens == 0:
        return float("inf")

    # Calculate final perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    print(f"Perplexity: {perplexity:.4f}")
    return perplexity
