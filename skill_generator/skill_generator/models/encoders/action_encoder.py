import torch
from torch import nn
from typing import Tuple


class ActionEncoder(nn.Module):
    def _sample(self, mu: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, acts: torch.Tensor, seq_l: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError
