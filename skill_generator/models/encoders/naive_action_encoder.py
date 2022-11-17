import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple
from .action_encoder import ActionEncoder
from ..utils.rnn import rnn_decoder, lstm_decoder, gru_decoder  # import for eval


class NaiveActionEncoder(ActionEncoder):
    def __init__(
            self,
            act_dim: int,
            latent_dim: int,
            layer_size: int,
            num_layers: int,
            rnn_model: str,
            policy_rnn_dropout_p: float
    ):
        super(NaiveActionEncoder, self).__init__()
        self.act_dim = act_dim
        self.rnn = eval(rnn_model)
        self.rnn = self.rnn(in_features=act_dim, hidden_size=layer_size, num_layers=num_layers, policy_rnn_dropout_p=policy_rnn_dropout_p)
        self.z_mu = nn.Linear(layer_size, latent_dim)
        self.z_scale = nn.Sequential(
            nn.Linear(layer_size, latent_dim),
            nn.Softplus(),
        )

    def _sample(self, mu: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        eps = torch.randn(*mu.size()).to(mu)
        return mu + scale * eps

    def forward(self, acts: torch.Tensor, seq_l: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param acts: action sequences with shape (B, T, N_a)
        :param seq_l: the length of the sequences in the batch
        :return: skill embeddings with shape (B, N_z), hidden state, and cell state
        """
        x = acts
        B, T, _ = x.shape
        x_packed = pack_padded_sequence(x, seq_l.cpu(), batch_first=True, enforce_sorted=False)

        x_packed, _ = self.rnn(x_packed)

        x, lens = pad_packed_sequence(x_packed, batch_first=True, total_length=T)
        x = x[torch.arange(B), seq_l - 1, :]  # we only need the last hidden state

        x_mu = self.z_mu(x)
        x_scale = self.z_scale(x)

        return self._sample(x_mu, x_scale), x_mu, x_scale
