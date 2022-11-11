import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS, STEP_OUTPUT
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union
import torch.distributions as D

from skill_generator.utils.distributions import State, ContState, Distribution


class SkillGenerator(pl.LightningModule):
    def __init__(
            self,
            action_encoder: DictConfig,
            action_decoder: DictConfig,
            prior_locator: DictConfig,
            optimizer: DictConfig,
            lr_scheduler: DictConfig,
            kl_beta: float,
            skill_dim: int,
            skill_len: int
    ):
        super().__init__()
        self.encoder = hydra.utils.instantiate(action_encoder)
        self.decoder = hydra.utils.instantiate(action_decoder)
        self.prior_locator = hydra.utils.instantiate(prior_locator)
        self.optimizer_config = optimizer
        self.lr_scheduler = lr_scheduler
        self.seq_l = torch.tensor(skill_len)
        self.kl_beta = kl_beta
        self.skill_dim = skill_dim
        self.dist = Distribution(dist='continuous')
        self.save_hyperparameters()

    def forward(self, acts, seq_l, robot_obs):
        B, _, _ = acts.shape
        z, z_mu, z_scale = self.encoder(acts, seq_l)
        loss, rec_acts = self.decoder.loss_and_acts(z, seq_l, acts, robot_obs)
        prior_locs = self.prior_locator(B)
        ret = {
            'rec_acts': rec_acts,
            'rec_loss': loss,
            'p_mu': prior_locs['p_mu'],
            'p_scale': prior_locs['p_scale'],
            'z_mu': z_mu,
            'z_scale': z_scale
        }
        return ret

    def std_normal(self, B):
        return ContState(torch.zeros((B, self.skill_dim), device=self.device, dtype=torch.float),
                         torch.ones((B, self.skill_dim), device=self.device, dtype=torch.float))

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer_config, params=self.parameters())
        scheduler = hydra.utils.instantiate(self.lr_scheduler, optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def compute_kl_loss(self, post_state: State, prior_state: State, mode=0):
        """
        Args:
            post_state: the state of posterior distribution
            prior_state: the state of prior distribution
            mode: 0 - no detach; 1 - detach prior; 2 - detach posterior

        Returns:
            average kl loss
        """
        if mode == 1:
            prior_state = self.dist.detach_state(prior_state)
        elif mode == 2:
            post_state = self.dist.detach_state(post_state)

        prior_dist = self.dist.get_dist(prior_state)  # prior
        post_dist = self.dist.get_dist(post_state)  # posterior

        kl_loss = D.kl_divergence(post_dist, prior_dist)
        return kl_loss

    def skill_classifier(self, actions, eps=1e-6):
        energy = 0.
        gripper_energy = 0.
        _, T, _ = actions.shape
        for i in range(T):
            energy += abs(actions[:, i, :6])
        for i in range(T - 1):
            gripper_energy += abs(actions[:, i + 1, 6] - actions[:, i, 6])

        translation = (energy[:, 0] + energy[:, 1] + energy[:, 2]) / 3
        rotation = (energy[:, 3] + energy[:, 4] + energy[:, 5]) / 3
        gripper = gripper_energy
        translation /= 0.13
        rotation /= 0.45
        gripper /= 2.
        s = torch.exp(translation) + torch.exp(rotation) + torch.exp(gripper)
        return torch.stack([torch.exp(translation) / s, torch.exp(rotation) / s, torch.exp(gripper) / s], dim=1).to(self.device)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        actions = batch['actions']
        robot_obs = batch['state_info']['robot_obs']
        B, T, _ = actions.shape
        seq_l = self.seq_l.repeat(B)
        ret = self.forward(actions, seq_l, robot_obs)
        translation_prior_state = ContState(ret['p_mu'][:, 0], ret['p_scale'][:, 0])
        rotation_prior_state = ContState(ret['p_mu'][:, 1], ret['p_scale'][:, 1])
        grasp_prior_state = ContState(ret['p_mu'][:, 2], ret['p_scale'][:, 2])

        skill_state = ContState(ret['z_mu'], ret['z_scale'])
        rec_loss = ret['rec_loss']
        reg_loss = self.compute_kl_loss(skill_state, self.std_normal(B)).mean()

        skill_types = self.skill_classifier(actions)
        prior_train_loss = skill_types[:, 0] * self.compute_kl_loss(skill_state, translation_prior_state, mode=2) + \
                           skill_types[:, 1] * self.compute_kl_loss(skill_state, rotation_prior_state, mode=2) + \
                           skill_types[:, 2] * self.compute_kl_loss(skill_state, grasp_prior_state, mode=2)

        total_loss = rec_loss + self.kl_beta * reg_loss + prior_train_loss.mean()

        self.log("train/rec_loss", rec_loss)
        self.log("train/reg_loss", reg_loss.mean())
        self.log("train/prior_train_loss", prior_train_loss.mean())

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Optional[STEP_OUTPUT]:
        actions = batch['actions']
        robot_obs = batch['state_info']['robot_obs']
        B, T, _ = actions.shape
        seq_l = self.seq_l.repeat(B)
        ret = self.forward(actions, seq_l, robot_obs=robot_obs)
        translation_prior_state = ContState(ret['p_mu'][:, 0], ret['p_scale'][:, 0])
        rotation_prior_state = ContState(ret['p_mu'][:, 1], ret['p_scale'][:, 1])
        grasp_prior_state = ContState(ret['p_mu'][:, 2], ret['p_scale'][:, 2])

        skill_state = ContState(ret['z_mu'], ret['z_scale'])
        rec_loss = ret['rec_loss']
        reg_loss = self.compute_kl_loss(skill_state, self.std_normal(B)).mean()

        skill_types = self.skill_classifier(actions)
        prior_train_loss = skill_types[:, 0] * self.compute_kl_loss(skill_state, translation_prior_state, mode=2) \
                           + skill_types[:, 1] * self.compute_kl_loss(skill_state, rotation_prior_state, mode=2) \
                           + skill_types[:, 2] * self.compute_kl_loss(skill_state, grasp_prior_state, mode=2)

        total_loss = rec_loss + self.kl_beta * reg_loss

        self.log("val/total_loss", total_loss)
        self.log("val/prior_train_loss", prior_train_loss)
        return total_loss

