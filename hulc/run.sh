#!/bin/sh
python training.py trainer.gpus=-1 datamodule.root_data_dir=dataset/task_D_D datamodule/datasets=vision_lang_shm ~callbacks/rollout ~callbacks/rollout_lh ~callbacks/tsne_plot