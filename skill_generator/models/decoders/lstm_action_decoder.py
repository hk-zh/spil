import torch
from torch import nn
from typing import Tuple, Optional
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from skill_generator.models.decoders.action_decoder import ActionDecoder


class LSTMActionDecoder(ActionDecoder):
    def __init__(self,
                 act_dim: int,
                 latent_dim: int,
                 layer_size: int,
                 num_layers: int,
                 criterion: str,
                 gripper_control: bool
                 ):
        super().__init__()
        self.latent_dim = latent_dim
        self.layer_size = layer_size
        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=layer_size, batch_first=True, num_layers=num_layers)
        self.mlp = nn.Linear(self.layer_size, act_dim)
        self.criterion = getattr(nn, criterion)()
        self.gripper_control = gripper_control

    def forward(self, latent_skill: torch.Tensor, seq_l: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq_l:
            latent_skill:
        """
        max_l = torch.max(seq_l)
        x = latent_skill[:, None, :]
        x = x.tile((1, max_l, 1))
        B, T, _ = x.shape
        x_packed = pack_padded_sequence(x, seq_l.cpu(), batch_first=True, enforce_sorted=False)

        x_packed, _ = self.lstm(x_packed)

        x, lens = pad_packed_sequence(x_packed, batch_first=True, total_length=T)
        x = self.mlp(x)
        return x

    def loss(self, latent_skill: torch.Tensor, seq_l: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_skill:
            seq_l:
            actions:

        Returns:
        """
        pred_actions = self.forward(latent_skill, seq_l)
        return self.criterion(pred_actions, actions)

    def loss_and_acts(self, latent_skill: torch.Tensor, seq_l: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_skill:
            seq_l:
            actions:

        Returns:
        """
        pred_actions = self.forward(latent_skill, seq_l)
        return self.criterion(pred_actions, actions), pred_actions

    def acts(self, latent_skill: torch.Tensor, seq_l: torch.Tensor, robot_obs: Optional[torch.Tensor]) -> torch.Tensor:
        pred_actions = self(latent_skill, seq_l)
        return pred_actions