HULC Model
```bash
python training.py trainer.gpus=-1 datamodule.root_data_dir=dataset/task_D_D datamodule/datasets=vision_lang_shm ~callbacks/rollout ~callbacks/rollout_lh ~callbacks/tsne_plot model/action_decoder=hulc_default loss.default.kl_beta=0.01
```

```bash
python training.py trainer.gpus=-1 datamodule.root_data_dir=dataset/task_D_D datamodule/datasets=vision_lang_shm ~callbacks/rollout ~callbacks/rollout_lh ~callbacks/tsne_plot model/action_decoder=skill loss.default.kl_beta=0.0005
```

CUDA_VISIBLE_DEVICES=0 python training.py trainer.gpus=-1 datamodule.root_data_dir=dataset/task_D_D datamodule/datasets=vision_lang ~callbacks/rollout ~callbacks/rollout_lh ~callbacks/tsne_plot model/action_decoder=skill  loss.kl_beta=0.0005 hydra.run.dir=../runs/2022-11-19/12-38-07 model.action_decoder.sg_chk_path='./checkpoints/SKILL_GENERATOR'  model.distribution=continuous