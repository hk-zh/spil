#!/bin/sh
python training.py trainer.gpus=-1 datamodule.root_data_dir=dataset/task_D_D datamodule.datasets.batch_size=256 loss.kl_beta=7e-5 loss.kl_sigma=5e-4 model.prior_seeking_mode=0