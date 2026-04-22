import math
from typing import Callable, Union

import torch
import torch.nn.functional as F


def compute_loss(
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], preds: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    assert len(preds.shape) >= 2 and len(labels.shape) >= 1
    return loss_fn(preds.view(-1, preds.shape[-1]), labels.view(-1))


def compute_perplexity_from_entropy(entropy: Union[float, torch.Tensor]):
    return math.exp(entropy)


def compute_perplexity(preds: torch.Tensor, labels: torch.Tensor, labels_ignore_idx: int = -100) -> torch.Tensor:
    return torch.exp(
        F.cross_entropy(input=preds.view(-1, preds.shape[-1]), target=labels.view(-1), ignore_index=labels_ignore_idx)
    ).detach()

