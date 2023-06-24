#!/bin/sh
CUDA_VISIBLE_DEVICES=0 python training.py \
trainer.gpus=-1 \
~callbacks/rollout \
~callbacks/tsne_plot \
datamodule.root_data_dir=dataset/task_ABC_D \
datamodule/datasets=vision_lang_shm \
datamodule.datasets.lang_dataset.batch_size=32 \
datamodule.datasets.vision_dataset.batch_size=32 \
model.action_decoder.sg_chk_path='./checkpoints/SKILL_GENERATOR_2023-02-24_15-12-18' \
loss.kl_beta=0.0001 \
model.action_decoder.beta=0.005 \
loss.clip_auxiliary_loss_beta=0.01