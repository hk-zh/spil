#!/bin/sh
HYDRA_FULL_ERROR=1 \
CUDA_VISIBLE_DEVICES=0 python training.py \
trainer.gpus=-1 \
datamodule.root_data_dir=dataset/task_D_D \
datamodule.datasets.batch_size=256 \
loss.kl_beta=1e-5 \
loss.kl_sigma=1e-4 \
model.prior_seeking_mode=2 \
model.skill_dim=20 \
model.action_decoder.layer_size=256 \
model.action_encoder.layer_size=256