from typing import Optional, Tuple
import torch
import torch.nn as nn


class ActionDecoder(nn.Module):
    def loss(
        self,
        latent_skill: torch.Tensor,
        seq_l: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def acts(
            self,
            latent_skill: torch.Tensor,
            seq_l: torch.Tensor,
    ):
        raise NotImplementedError

    def loss_and_acts(
            self,
            latent_skill: torch.Tensor,
            seq_l: torch.Tensor,
            actions: torch.Tensor
    ):
        raise NotImplementedError

    def forward(
            self,
            latent_skill: torch.Tensor,
            seq_l: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError
