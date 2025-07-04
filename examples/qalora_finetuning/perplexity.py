# Directly taken from https://huggingface.co/spaces/evaluate-measurement/perplexity/blob/main/perplexity.py
# TODO replace with a strided version https://github.com/huggingface/transformers/issues/9648#issuecomment-812981524
import itertools

import torch
from datasets import Dataset, load_dataset
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
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, dataset: Dataset = None, batch_size=32, stride=512
):
    """
    Better perplexity calculation for causal language models.

    Args:
        model: A pretrained language model
        tokenizer: The tokenizer used to preprocess the data
        dataset: A dataset to calculate perplexity on. If None, the wikitext-2 test set is used.
        batch_size: The batch size to use for perplexity calculation
        stride: The stride to use for perplexity calculation - Important, changing this will change your results



    Comparison again other implementations:
    - https://huggingface.co/docs/transformers/perplexity - takes the mean of means giving it the wrong value
    -  https://github.com/huggingface/evaluate/blob/main/metrics/perplexity/perplexity.py - compelx and crops sentances so it's not comparable
    - https://github.com/ggerganov/llama.cpp/tree/master/examples/perplexity - good but in cpp
    - https://github.com/huggingface/transformers/issues/9648#issuecomment-812981524 - doesn't use special tokens

    Limitations of this implementation:
    - if a token is at the start of a strided window, it has no context, so it's perplexity is higher. TODO: have overlapping windows
    - uses special tokens, hard to compare to scores that do not

    """
    if dataset is None:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]
    else:
        dataset = dataset.filter(lambda x: len(x) > 0)

        dataset = dataset["text"]
    dataset = [x for x in dataset if len(x) > 0]
    device = "cuda:0"

    i = tokenizer(dataset, add_special_tokens=True, return_special_tokens_mask=True)
    input_ids = torch.tensor(list(itertools.chain(*i.input_ids))).to(torch.long).unsqueeze(0)

    # without padding or truncation we don't need attention but we do need special_tokens
    attention_mask = torch.tensor(list(itertools.chain(*i.attention_mask))).to(torch.bool).unsqueeze(0)
    special_tokens_mask = torch.tensor(list(itertools.chain(*i.special_tokens_mask))).to(torch.bool).unsqueeze(0)
    # let's not calc the perplexity on special_tokens
    loss_mask = attention_mask & ~special_tokens_mask

    text_len = input_ids.size(1)
    lls = []
    for i in tqdm(range(0, text_len, batch_size * stride)):
        b_input_ids, target_ids = create_batch(input_ids, loss_mask, i, batch_size, stride)

        # Skip empty batches
        if b_input_ids is None:
            continue

        b_input_ids = b_input_ids.to(device)
        target_ids = target_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(b_input_ids).logits
        log_likelihood = nll_loss_no_mean(logits, target_ids)
        lls.extend(log_likelihood.view(-1).cpu().tolist())

    lls = torch.tensor(lls)
    ppl = lls.mean().exp()
    return ppl.cpu().item()
