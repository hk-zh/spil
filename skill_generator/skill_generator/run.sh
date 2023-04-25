CUDA_VISIBLE_DEVICES=0 python training.py \
trainer.gpus=-1 \
datamodule.root_data_dir=dataset/task_D_D \
datamodule.datasets.batch_size=256 \
loss.kl_beta=1e-6 \
loss.kl_sigma=1e-5 \
model.prior_seeking_balance=0.9 \
model.skill_dim=20 \
model.max_skill_len=5 \
model.min_skill_len=2 \
model.action_decoder.layer_size=256 \
model.action_encoder.layer_size=256