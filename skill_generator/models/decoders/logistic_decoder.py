import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from skill_generator.models.decoders.action_decoder import ActionDecoder
from ..utils.rnn import rnn_decoder, lstm_decoder, gru_decoder  # import for eval
from torch.distributions.normal import Normal


class View(nn.Module):
    def __init__(self, act_dim, n_dist):
        super().__init__()
        self.act_dim = act_dim
        self.n_dict = n_dist

    def forward(self, x):
        return x.view(*x.shape[:-1], self.act_dim, self.n_dict)


class LogisticDecoder(ActionDecoder):
    def __init__(self,
                 act_dim: int,
                 latent_dim: int,
                 layer_size: int,
                 num_layers: int,
                 criterion: str,
                 rnn_model: str,
                 n_dist: int,
                 policy_rnn_dropout_p: float
                 ):
        super(LogisticDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.layer_size = layer_size
        self.act_dim = act_dim
        self.n_dist = n_dist
        self.rnn = eval(rnn_model)
        self.rnn = self.rnn(in_features=latent_dim, hidden_size=layer_size, num_layers=num_layers, policy_rnn_dropout_p=policy_rnn_dropout_p)
        self.mu = nn.Sequential(
            nn.Linear(self.layer_size, act_dim * n_dist),
            View(act_dim, n_dist)
        )
        self.scale = nn.Sequential(
            nn.Linear(self.layer_size, act_dim * n_dist),
            nn.Softplus(),
            View(act_dim, n_dist),
        )
        self.prob = nn.Sequential(
            nn.Linear(self.layer_size, act_dim * n_dist),
            View(act_dim, n_dist),
            nn.Softmax(dim=-1)
        )
        self.criterion = getattr(nn, criterion)()

    @staticmethod
    def _sample(prob, mu, scale):
        eps = torch.randn(*mu.size()).to(mu)
        return torch.mean((mu + scale * eps) * prob, dim=-1)

    def _loss(self, prob, mu, scale, actions):
        dist = Normal(mu, scale)
        actions = actions[:, :, :, None]
        actions = actions.tile((1, 1, 1, self.n_dist))
        return torch.mean(-1 * prob * dist.log_prob(actions))

    def loss(
        self,
        latent_skill: torch.Tensor,
        seq_l: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        prob, mu, scale = self(latent_skill, seq_l)
        return self._loss(prob, mu, scale, actions)

    def loss_and_acts(
            self,
            latent_skill: torch.Tensor,
            seq_l: torch.Tensor,
            actions: torch.Tensor
    ):
        prob, mu, scale = self(latent_skill, seq_l)
        pred_actions = self._sample(prob, mu, scale)
        loss = self._loss(prob, mu, scale, actions)
        return loss, pred_actions

    def acts(
            self,
            latent_skill: torch.Tensor,
            seq_l: torch.Tensor,
    ):
        prob, mu, scale = self(latent_skill, seq_l)
        pred_actions = self._sample(prob, mu, scale)
        return pred_actions

    def forward(
            self,
            latent_skill: torch.Tensor,
            seq_l: torch.Tensor,
    ) -> torch.Tensor:
        max_l = torch.max(seq_l)
        x = latent_skill[:, None, :]
        x = x.tile((1, max_l, 1))
        B, T, _ = x.shape
        x_packed = pack_padded_sequence(x, seq_l.cpu(), batch_first=True, enforce_sorted=False)

        x_packed, _ = self.rnn(x_packed)

        x, lens = pad_packed_sequence(x_packed, batch_first=True, total_length=T)

        prob = self.prob(x)
        mu = self.mu(x)
        scale = self.scale(x)

        return prob, mu, scale


