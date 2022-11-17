import torch
from torch import nn
from typing import Tuple, Optional
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from skill_generator.models.decoders.action_decoder import ActionDecoder
from ..utils.rnn import rnn_decoder, lstm_decoder, gru_decoder  # import for eval


class DeterministicDecoder(ActionDecoder):
    def __init__(self,
                 act_dim: int,
                 latent_dim: int,
                 layer_size: int,
                 num_layers: int,
                 criterion: str,
                 rnn_model: str,
                 policy_rnn_dropout_p: float
                 ):
        super().__init__()
        self.latent_dim = latent_dim
        self.layer_size = layer_size
        self.rnn = eval(rnn_model)
        self.rnn = self.rnn(in_features=latent_dim, hidden_size=layer_size, num_layers=num_layers, policy_rnn_dropout_p=policy_rnn_dropout_p)
        self.mlp = nn.Linear(self.layer_size, act_dim)
        self.criterion = getattr(nn, criterion)()

    @staticmethod
    def _hinge_loss(pred_gripper_actions, gt_gripper_actions, eps=0.2):
        return torch.clamp(1.0 - pred_gripper_actions * gt_gripper_actions, min=eps).mean()

    def _loss(self, pred_actions, gt_actions):
        loss = self.criterion(pred_actions[..., :6], gt_actions[..., :6])
        hinge_loss = self._hinge_loss(pred_actions[..., 6], gt_actions[..., 6])
        return (loss + hinge_loss) / 2.

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

        x_packed, _ = self.rnn(x_packed)

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
        loss = self._loss(pred_actions, actions)
        return loss

    def loss_and_acts(self, latent_skill: torch.Tensor, seq_l: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_skill:
            seq_l:
            actions:
        Returns:
        """
        pred_actions = self.forward(latent_skill, seq_l)
        loss = self._loss(pred_actions, actions)
        return loss, pred_actions

    def acts(self, latent_skill: torch.Tensor, seq_l: torch.Tensor) -> torch.Tensor:
        pred_actions = self(latent_skill, seq_l)
        return pred_actions
