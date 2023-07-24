import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Dict, List
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.trainer.supporters import CombinedLoader
from pathlib import Path
import numpy as np
from typing import Dict, Optional
import skill_generator
import hydra


class SGDataModule(pl.LightningDataModule):
    def __init__(
            self,
            datasets: DictConfig,
            root_data_dir: str = "data",
            num_workers: int = 8,
            shuffle_val: bool = False,
            **kwargs: Dict,
    ):
        super().__init__()
        self.datasets_cfg = datasets
        self.train_dataset = None
        self.val_dataset = None
        self.train_sampler = None
        self.val_sampler = None
        self.num_workers = num_workers
        root_data_path = Path(root_data_dir)
        if not root_data_path.is_absolute():
            root_data_path = Path(skill_generator.__file__).parent.parent.parent / root_data_path
        self.training_dir = root_data_path / "training"
        self.val_dir = root_data_path / "validation"
        self.shuffle_val = shuffle_val

    def prepare_data(self) -> None:
        # check if files already exist
        dataset_exist = np.any([len(list(self.training_dir.glob(extension))) for extension in ["*.npz", "*.pkl"]])
        if not dataset_exist:
            print(f"No dataset found in {self.training_dir}.")
            exit()

    def setup(self, stage: Optional[str] = None) -> None:
        train_dataset = hydra.utils.instantiate(
            self.datasets_cfg, datasets_dir=self.training_dir
        )
        val_dataset = hydra.utils.instantiate(self.datasets_cfg, datasets_dir=self.val_dir)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_dataset.batch_size,
                          num_workers=self.train_dataset.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_dataset.batch_size,
                          num_workers=self.val_dataset.num_workers, pin_memory=True)
