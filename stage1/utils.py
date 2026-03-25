import torch
from typing import List


def _set_requires_grad(params, value: bool):
    for p in params:
        p.requires_grad = value


def _freeze_llm(model):
    _set_requires_grad(model.language_model.parameters(), False)


def _unfreeze_image_encoder(model, compute_dtype, device):
    model.vision_tower.to(dtype=compute_dtype, device=device)
    _set_requires_grad(model.vision_tower.parameters(), False)
    _set_requires_grad(model.multi_modal_projector.parameters(), True)


def _count_params(model):
    total = trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def _log(*args):
    print("[Stage1]", *args, flush=True)


def _print_trainable_parameters(model):
    trainable, total = _count_params(model)
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    _log(f"trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    _log(f"trainable modules (first 20): {trainable_names[:20]}")



# dataset.py
def _pad_sequence(sequences: List[torch.Tensor], padding_value: int = 0) -> torch.Tensor:
    """右側 padding，回傳 [batch, max_len] tensor。"""
    max_len = max(s.size(0) for s in sequences)
    batch = sequences[0].new_full((len(sequences), max_len), padding_value)
    for i, seq in enumerate(sequences):
        batch[i, :seq.size(0)] = seq
    return batch