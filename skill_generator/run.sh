#!/bin/sh
python training.py trainer.gpus=-1 datamodule.root_data_dir=dataset/task_D_D loss.kl_beta=0.001