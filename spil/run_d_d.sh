#!/bin/sh
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0 python training.py \
trainer.gpus=-1 \
datamodule.root_data_dir=dataset/task_D_D \
datamodule/datasets=vision_lang_shm \
datamodule.datasets.lang_dataset.batch_size=32 \
datamodule.datasets.vision_dataset.batch_size=32 \
model/action_decoder=skill \
model.action_decoder.perceptual_emb_slice.0=0 \
model.action_decoder.out_features=20 \
model.action_decoder.skill_len=5 \
model.action_decoder.sg_chk_path='./checkpoints/SKILL_GENERATOR_2023-06-06_23-03-14' \
model/distribution=discrete \
loss.kl_beta=0.0005 \
loss.clip_auxiliary_loss_beta=0.01 \
model.action_decoder.gamma_1=0.005 \
model.action_decoder.gamma_2=0.00001 \
hydra.run.dir='../runs/2023-06-15/09-30-13'
