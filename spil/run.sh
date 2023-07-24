#!/bin/sh
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python training.py \
trainer.gpus=-1 \
datamodule.root_data_dir=dataset/task_ABC_D \
datamodule/datasets=vision_lang_shm \
model.action_decoder.sg_chk_path=./checkpoints/SKILL_GENERATOR_2023-02-24_15-12-18 \
