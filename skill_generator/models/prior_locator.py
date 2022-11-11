import torch
from torch import nn


class PriorLocator(nn.Module):
    def __init__(self, base_skills_num, p_hidden_dim, latent_dim):
        super().__init__()
        self.prior_mu_extractor = nn.Sequential(
            nn.Linear(base_skills_num, p_hidden_dim),
            nn.ReLU(),
            nn.Linear(p_hidden_dim, p_hidden_dim),
            nn.ReLU(),
            nn.Linear(p_hidden_dim, p_hidden_dim),
            nn.ReLU(),
            nn.Linear(p_hidden_dim, p_hidden_dim),
            nn.ReLU(),
            nn.Linear(p_hidden_dim, p_hidden_dim),
            nn.ReLU(),
            nn.Linear(p_hidden_dim, latent_dim),
        )

        self.prior_scale_extractor = nn.Sequential(
            nn.Linear(base_skills_num, p_hidden_dim),
            nn.ReLU(),
            nn.Linear(p_hidden_dim, p_hidden_dim),
            nn.ReLU(),
            nn.Linear(p_hidden_dim, p_hidden_dim),
            nn.ReLU(),
            nn.Linear(p_hidden_dim, p_hidden_dim),
            nn.ReLU(),
            nn.Linear(p_hidden_dim, p_hidden_dim),
            nn.ReLU(),
            nn.Linear(p_hidden_dim, latent_dim),
            nn.Softplus(),
        )
        self.register_buffer('one_hot_keys', torch.eye(base_skills_num))

    def forward(self, repeat):
        p_mu = self.prior_mu_extractor(self.one_hot_keys)
        p_scale = self.prior_scale_extractor(self.one_hot_keys)

        p_mu = p_mu[None, ...]
        p_scale = p_scale[None, ...]

        p_mu = torch.tile(p_mu, (repeat, 1, 1))
        p_scale = torch.tile(p_scale, (repeat, 1, 1))

        ret = {
            'p_mu': p_mu,
            'p_scale': p_scale
        }
        return ret
