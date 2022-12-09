from MulticoreTSNE import MulticoreTSNE as TSNE
import numpy as np
from PIL import Image
from pytorch_lightning import Callback
from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import plotly.express as px
import io
import torch


def plotly_fig2array(fig):
    """convert Plotly fig to  an array"""
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)


def log_figure(fig, logger, step, name):
    if isinstance(logger, WandbLogger):
        logger.experiment.log({name: fig})
    else:
        logger.experiment.add_image(name, plotly_fig2array(fig), global_step=step)


def create_tsne_figure(x_tsne, labels, step, logger, name, opacity=0.3):
    fig = px.scatter(
        x=x_tsne[:, 0].flatten(),
        y=x_tsne[:, 1].flatten(),
        color=labels,
        opacity=opacity
    )
    log_figure(fig, logger, step, name)


class TSNEPlot(Callback):
    def __init__(self, perplexity, n_jobs, plot_percentage, opacity, marker_size):
        self.perplexity = perplexity
        self.n_jobs = n_jobs
        self.plot_percentage = plot_percentage
        self.opacity = opacity
        self.marker_size = marker_size
        self.sampled_latent_skills = []
        self.sampled_skill_types = []

    def _get_tsne(self, sampled_plans):
        x_tsne = TSNE(perplexity=self.perplexity, n_jobs=self.n_jobs).fit_transform(sampled_plans.cpu())
        return x_tsne

    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: Optional[STEP_OUTPUT],
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        self.sampled_latent_skills.append(outputs["latent_skills"])
        self.sampled_skill_types.append(outputs["skill_types"])

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if pl_module.global_step > 0:  # ignore the initial step because it doesn't contain any information yet
            sampled_latent_skills = torch.cat(self.sampled_latent_skills)
            sampled_skill_types = torch.cat(self.sampled_skill_types)
            self.sampled_skill_types = []
            self.sampled_latent_skills = []
            sampled_latent_skills = pl_module.all_gather(sampled_latent_skills).squeeze()
            sampled_skill_types = pl_module.all_gather(sampled_skill_types).squeeze()
            sampled_skill_types = sampled_skill_types.cpu().numpy()

            n = sampled_skill_types.shape[0]
            ids = np.random.choice(n, replace=False, size=int(n * self.plot_percentage))
            sampled_latent_skills = sampled_latent_skills[ids]
            sampled_skill_types = sampled_skill_types[ids]

            x_tsne = self._get_tsne(sampled_latent_skills)
            create_tsne_figure(x_tsne, sampled_skill_types, pl_module.global_step, logger=pl_module.logger, name="skill_latent_space")
