#!/bin/sh
CUDA_VISIBLE_DEVICES=0 python training.py \
trainer.gpus=-1 \
datamodule.root_data_dir=dataset/task_D_D \
datamodule.datasets.batch_size=256 \
loss.kl_beta=2e-6 \
loss.kl_sigma=5e-5 \
model.prior_seeking_mode=0 \
model.skill_dim=18 \
model.action_decoder.layer_size=256 \
model.action_encoder.layer_size=256