#!/bin/sh
CUDA_VISIBLE_DEVICES=0 python training.py \
trainer.gpus=-1 \
datamodule.root_data_dir=dataset/task_ABC_D \
datamodule/datasets=vision_lang_shm \
~callbacks/rollout \
~callbacks/tsne_plot \
model/action_decoder=skill \
model.action_decoder.perceptual_emb_slice.0=0 \
model.action_decoder.out_features=20 \
model.action_decoder.skill_len=5 \
model.action_decoder.sg_chk_path=./checkpoints/SKILL_GENERATOR_2023-02-24_15-12-18 \
model/distribution=discrete \
loss.kl_beta=0.0001 \
loss.clip_auxiliary_loss_beta=0.01
