from typing import Optional, Tuple
import torch
import torch.nn as nn


class ActionDecoder(nn.Module):
    def act(
        self,
        latent_skill: torch.Tensor,
        seq_l: torch.Tensor,
        robot_obs: Optional[torch.Tensor]
    ) -> torch.Tensor:
        raise NotImplementedError

    def loss(
        self,
        latent_skill: torch.Tensor,
        seq_l: torch.Tensor,
        actions: torch.Tensor,
        robot_obs: Optional[torch.Tensor]
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
            self,
            latent_skill: torch.Tensor,
            seq_l: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError
