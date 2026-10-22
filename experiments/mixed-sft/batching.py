"""Completion-only SFT batches; no model loading, truncation, or optimizer steps."""

import torch
import torch.nn.functional as F


def collate_completion_batch(items, pad_token_id, device="cpu"):
    """Collate 1..4 (episode_id, prompt_ids, target_ids) rows; targets include EOS.

    Left padding puts every completion at the right edge. ``labels`` cover only
    the last max_target_length positions, with -100 before shorter completions.
    Mask construction uses sequence lengths, so EOS may also be the pad token.
    """
    items = list(items)
    if not 1 <= len(items) <= 4:
        raise ValueError("A completion batch must contain 1..4 examples")
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("pad_token_id must be a nonnegative integer")
    rows = []
    for episode_id, prompt_ids, target_ids in items:
        prompt_ids, target_ids = list(prompt_ids), list(target_ids)
        if not prompt_ids or not target_ids:
            raise ValueError("Every example needs a prompt and an EOS-inclusive target")
        if any(type(token) is not int or token < 0 for token in prompt_ids + target_ids):
            raise ValueError("Token IDs must be nonnegative integers")
        rows.append((episode_id, prompt_ids, target_ids))
    width = max(len(prompt) + len(target) for _, prompt, target in rows)
    target_width = max(len(target) for _, _, target in rows)
    ids, masks, labels = [], [], []
    for _, prompt, target in rows:
        padding = width - len(prompt) - len(target)
        ids.append([pad_token_id] * padding + prompt + target)
        masks.append([0] * padding + [1] * (len(prompt) + len(target)))
        labels.append([-100] * (target_width - len(target)) + target)
    attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
    position_ids = (attention_mask.cumsum(-1) - 1).clamp_min(0)
    return {
        "episode_ids": [row[0] for row in rows],
        "input_ids": torch.tensor(ids, dtype=torch.long, device=device),
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "target_lengths": torch.tensor([len(row[2]) for row in rows], device=device),
        "logits_to_keep": target_width + 1,
    }


def completion_loss(model, batch):
    """Mean of each example's mean target-token loss, preserving batch1 weighting.

    The retained logits start one position before the label suffix. Dropping the
    final logit performs the causal shift; prompt and pad positions have no loss.
    Explicit positions keep real tokens numbered from zero after left padding.
    """
    logits = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        use_cache=False,
        logits_to_keep=batch["logits_to_keep"],
    ).logits
    labels = batch["labels"]
    expected = (labels.shape[0], labels.shape[1] + 1)
    if logits.ndim != 3 or tuple(logits.shape[:2]) != expected:
        raise ValueError("Model logits do not match the requested completion suffix")
    shifted_logits = logits[:, :-1, :]
    token_losses = F.cross_entropy(
        shifted_logits.reshape(-1, logits.shape[-1]).float(),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(labels)
    per_example = token_losses.sum(-1) / batch["target_lengths"]
    return per_example.mean()
