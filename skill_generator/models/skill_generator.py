import torch
from torch import nn
import hydra
from omegaconf import DictConfig
from skill_generator.models.encoders.lstm_action_encoder import LSTMActionEncoder
from skill_generator.models.decoders.lstm_action_decoder import LSTMActionDecoder


class skill_generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.act_dim = config.act_dim
        self.latent_dim = config.latent_dim
        self.encoder = LSTMActionEncoder(act_dim=self.act_dim, latent_dim=self.latent_dim)
        self.decoder = LSTMActionDecoder(act_dim = self.act_dim, latent_dim=self.latent_dim)
        self.start_symbol = torch.tensor(config.start_symbol)
        self.p_hidden_dim = 128
        self.prior_mu_extractor = nn.Sequential(
            nn.Linear(config.base_skills_num, self.p_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.p_hidden_dim, self.p_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.p_hidden_dim, self.p_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.p_hidden_dim, self.p_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.p_hidden_dim, self.p_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.p_hidden_dim, self.latent_dim),
        )

        self.prior_scale_extractor = nn.Sequential(
            nn.Linear(config.base_skills_num, self.p_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.p_hidden_dim, self.p_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.p_hidden_dim, self.p_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.p_hidden_dim, self.p_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.p_hidden_dim, self.p_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.p_hidden_dim, self.latent_dim),
            nn.Softplus(),
        )

    def forward(self, acts, seq_l, one_hot_keys):
        z, z_mu, z_scale = self.encoder(acts, seq_l)
        rec_acts = self.decoder(seq_l, z)
        p_mu = self.prior_mu_extractor(one_hot_keys)
        p_scale = self.prior_scale_extractor(one_hot_keys)
        ret = {
            'rec_acts': rec_acts,
            'p_mu': p_mu,
            'p_scale': p_scale,
            'z_mu': z_mu,
            'z_scale': z_scale
        }
        return ret