## Some Useful Commands

```bash
python training.py trainer.gpus=-1 datamodule.root_data_dir=dataset/task_D_D datamodule.datasets.batch_size=128 loss.kl_beta=1e-4 loss.kl_sigma=1e-2 model.prior_seeking_mode=2
```

```bash
python training.py trainer.gpus=-1 datamodule.root_data_dir=dataset/task_D_D datamodule.datasets.batch_size=128 loss.kl_beta=7e-5 loss.kl_sigma=1e-4 model.prior_seeking_mode=0
```

```bash
python training.py trainer.gpus=-1 datamodule.root_data_dir=dataset/task_D_D datamodule.datasets.batch_size=128 loss.kl_beta=2e-4 loss.kl_sigma=1e-2 model.prior_seeking_mode=2 model.scale=[1.28, 0.94, 2.0]
```