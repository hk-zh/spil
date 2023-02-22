import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS, STEP_OUTPUT
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union
import torch.distributions as D

from skill_generator.utils.distributions import State, ContState, Distribution
from hulc.models.decoders.utils.gripper_control import tcp_to_world_frame, world_to_tcp_frame
import torch.nn.functional as F


class SkillGenerator(pl.LightningModule):
    def __init__(
            self,
            action_encoder: DictConfig,
            action_decoder: DictConfig,
            prior_locator: DictConfig,
            optimizer: DictConfig,
            lr_scheduler: DictConfig,
            kl_beta: float,
            kl_sigma: float,
            prior_seeking_balance: int,
            skill_dim: int,
            min_skill_len: int,
            max_skill_len: int,
            magic_scale: Tuple,
            prior_locator_weight: Tuple
    ):
        super().__init__()
        self.encoder = hydra.utils.instantiate(action_encoder)
        self.decoder = hydra.utils.instantiate(action_decoder)
        self.prior_locator = hydra.utils.instantiate(prior_locator)
        self.optimizer_config = optimizer
        self.lr_scheduler = lr_scheduler
        self.seq_l = torch.tensor(max_skill_len)
        self.kl_beta = kl_beta
        self.kl_sigma = kl_sigma
        self.balance = prior_seeking_balance
        self.skill_dim = skill_dim
        self.dist = Distribution(dist='continuous')
        self.scale = magic_scale
        self.pl_w = prior_locator_weight
        self.save_hyperparameters()

    def forward(self, acts, seq_l):
        B, _, _ = acts.shape
        z, z_mu, z_scale = self.encoder(acts, seq_l)
        loss, rec_acts = self.decoder.loss_and_acts(z, seq_l, acts)
        prior_locs = self.prior_locator(B)
        ret = {
            'rec_acts': rec_acts,
            'rec_loss': loss,
            'p_mu': prior_locs['p_mu'],
            'p_scale': prior_locs['p_scale'],
            'z_mu': z_mu,
            'z_scale': z_scale,
            'z': z
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

    def compute_kl_loss(self, post_state: State, prior_state: State, balance=0.8):
        """
        Args:
            post_state: the state of posterior distribution
            prior_state: the state of prior distribution
            balance: balance for approaching

        Returns:
            averaged kl loss
        """
        assert 1.0 >= balance >= 0.0

        prior_state_detached = self.dist.detach_state(prior_state)
        post_state_detached = self.dist.detach_state(post_state)

        prior_dist = self.dist.get_dist(prior_state)  # prior
        post_dist = self.dist.get_dist(post_state)  # posterior

        prior_dist_detached = self.dist.get_dist(prior_state_detached)  # prior
        post_dist_detached = self.dist.get_dist(post_state_detached)  # posterior

        kl_loss_0 = D.kl_divergence(post_dist, prior_dist_detached)
        kl_loss_1 = D.kl_divergence(post_dist_detached, prior_dist)
        return balance * kl_loss_0 + (1-balance) * kl_loss_1

    def skill_classifier(self, actions, eps=0.05):
        gripper_energy = 0.
        _, T, _ = actions.shape
        energy = torch.sum(torch.abs(actions[:, :, :6]), dim=1)
        for i in range(T - 1):
            gripper_energy += abs(actions[:, i + 1, 6] - actions[:, i, 6])

        translation = torch.norm(energy[:, :3], dim=1)
        rotation = torch.norm(energy[:, 3:6], dim=1)
        gripper = gripper_energy

        translation /= self.scale[0]
        rotation /= self.scale[1]
        gripper /= self.scale[2]
        rotation[rotation < eps] = -1.

        t = torch.stack([translation, rotation, gripper], dim=-1)
        B, _ = t.shape
        skill_types = torch.argmax(t, dim=-1)
        one_hot_key = torch.zeros_like(t)
        one_hot_key[torch.arange(B), skill_types] = 1.
        return {
            'one_hot_key': one_hot_key,
            'skill_types': skill_types
        }

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        actions = batch['actions']
        robot_obs = batch['state_info']['robot_obs']
        tcp_actions = world_to_tcp_frame(actions, robot_obs=robot_obs)
        B, T, _ = actions.shape
        seq_l = self.seq_l.repeat(B)
        ret = self.forward(tcp_actions, seq_l)
        translation_prior_state = ContState(ret['p_mu'][:, 0], ret['p_scale'][:, 0])
        rotation_prior_state = ContState(ret['p_mu'][:, 1], ret['p_scale'][:, 1])
        grasp_prior_state = ContState(ret['p_mu'][:, 2], ret['p_scale'][:, 2])

        skill_state = ContState(ret['z_mu'], ret['z_scale'])
        rec_loss = ret['rec_loss']
        reg_loss = self.compute_kl_loss(skill_state, self.std_normal(B)).mean()

        sc_ret = self.skill_classifier(tcp_actions)
        ohk, _ = sc_ret['one_hot_key'], sc_ret['skill_types']
        prior_train_loss = (self.pl_w[0] * ohk[:, 0] * self.compute_kl_loss(skill_state, translation_prior_state,
                                                                            balance=self.balance) +
                            self.pl_w[1] * ohk[:, 1] * self.compute_kl_loss(skill_state, rotation_prior_state,
                                                                            balance=self.balance) +
                            self.pl_w[2] * ohk[:, 2] * self.compute_kl_loss(skill_state, grasp_prior_state,
                                                                            balance=self.balance)).mean()

        total_loss = rec_loss + self.kl_beta * reg_loss + self.kl_sigma * prior_train_loss

        self.log("train/rec_loss", rec_loss)
        self.log("train/reg_loss", reg_loss.mean())
        self.log("train/prior_train_loss", prior_train_loss)

        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Optional[STEP_OUTPUT]:
        output = {}
        actions = batch['actions']
        robot_obs = batch['state_info']['robot_obs']
        tcp_actions = world_to_tcp_frame(actions, robot_obs=robot_obs)

        B, T, _ = actions.shape
        seq_l = self.seq_l.repeat(B)
        ret = self.forward(tcp_actions, seq_l)
        translation_prior_state = ContState(ret['p_mu'][:, 0], ret['p_scale'][:, 0])
        rotation_prior_state = ContState(ret['p_mu'][:, 1], ret['p_scale'][:, 1])
        grasp_prior_state = ContState(ret['p_mu'][:, 2], ret['p_scale'][:, 2])

        skill_state = ContState(ret['z_mu'], ret['z_scale'])
        rec_loss = ret['rec_loss']

        reg_loss = self.compute_kl_loss(skill_state, self.std_normal(B)).mean()

        sc_ret = self.skill_classifier(tcp_actions)
        ohk, skill_types = sc_ret['one_hot_key'], sc_ret['skill_types']
        prior_train_loss = (
                self.pl_w[0] * ohk[:, 0] * self.compute_kl_loss(skill_state, translation_prior_state)
                + self.pl_w[1] * ohk[:, 1] * self.compute_kl_loss(skill_state, rotation_prior_state)
                + self.pl_w[2] * ohk[:, 2] * self.compute_kl_loss(skill_state, grasp_prior_state)).mean()

        total_loss = rec_loss + self.kl_beta * reg_loss + self.kl_sigma * prior_train_loss

        rec_acts = ret['rec_acts']
        mae_trans_loss = F.l1_loss(rec_acts[..., :3], tcp_actions[..., :3])
        mae_rot_loss = F.l1_loss(rec_acts[..., 3:6], tcp_actions[..., 3:6])
        max_grp_loss = F.l1_loss(rec_acts[..., 6], tcp_actions[..., 6])

        self.log("val/total_loss", total_loss)
        self.log("val/prior_train_loss", prior_train_loss)
        self.log("val/mae_trans_loss", mae_trans_loss)
        self.log("val/mae_rot_loss", mae_rot_loss)
        self.log("val/mae_grp_loss", max_grp_loss)

        output["latent_skills"] = ret['z']
        output["skill_types"] = skill_types

        return output
