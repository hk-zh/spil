# PSIL
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[<b>Skill Priors to Increase Generalisation Abilities for Language-Conditioned Robot Manipulation under Unstructured Data</b>](https://arxiv.org/pdf/2204.06252.pdf)

![](media/hulc_rollout.gif)
## Installation
To begin, clone this repository locally
```bash
git clone --recurse-submodules https://github.com/Hongkuan-Zhou/spil
export ROOT=$(pwd)/hulc

```
Install requirements:
```bash
cd $ROOT
conda create -n spil_venv python=3.8  # or use virtualenv
conda activate spil_venv
sh install.sh
```
If you encounter problems installing pyhash, you might have to downgrade setuptools to a version below 58.

## Download
### CALVIN Dataset
If you want to train on the [CALVIN](https://github.com/mees/calvin) dataset, choose a split with:
```bash
cd $ROOT/dataset
sh download_data.sh D | ABC | ABCD | debug
```
If you want to get started without downloading the whole dataset, use the argument `debug` to download a small debug dataset (1.3 GB).
### Language Embeddings
We provide the precomputed embeddings of the different Language Models we evaluate in the paper.
The script assumes the corresponding split has been already downloaded.
```bash
cd $ROOT/dataset
sh download_lang_embeddings.sh D | ABC | ABCD
```

### Pre-trained Models
on the way...
## Hardware Requirements

Trained with:
- **GPU** - 1x NVIDIA Tesla V100 16GB
- **RAM** - 256GB
- **OS** - Ubuntu 20.04

## Training
To train the model with the maximum amount of available GPUS, run:
```
python hulc/training.py trainer.gpus=-1 datamodule.root_data_dir=path/to/dataset datamodule/datasets=vision_lang_shm
```
The `vision_lang_shm` option loads the CALVIN dataset into shared memory at the beginning of the training,
speeding up the data loading during training.
The preparation of the shared memory cache will take some time
(approx. 20 min at our SLURM cluster). \
If you want to use the original data loader (e.g. for debugging) just override the command with `datamodule/datasets=vision_lang`. \
For an additional speed up, you can disable the evaluation callbacks during training by adding `~callbacks/rollout` and `~callbacks/rollout_lh`


### Ablations
Multi-context imitation learning (MCIL), (Lynch et al., 2019):
```
python hulc/training.py trainer.gpus=-1 datamodule.root_data_dir=path/to/dataset datamodule/datasets=vision_lang_shm model=mcil
datamodule=mcil
```

Goal-conditioned behavior cloning (GCBC), (Lynch et al., 2019):
```
python hulc/training.py trainer.gpus=-1 datamodule.root_data_dir=path/to/dataset datamodule/datasets=vision_lang_shm model=gcbc
~callbacks/tsne_plot
```


## Evaluation
See detailed inference instructions on the [CALVIN repo](https://github.com/mees/calvin#muscle-evaluation-the-calvin-challenge).
```
python hulc/evaluation/evaluate_policy.py --dataset_path <PATH/TO/DATASET> --train_folder <PATH/TO/TRAINING/FOLDER>
```
Set `--train_folder $HULC_ROOT/checkpoints/HULC_D_D` to evaluate our [pre-trained models](#pre-trained-models).

Optional arguments:

- `--checkpoint <PATH/TO/CHECKPOINT>`: by default, the evaluation loads the last checkpoint in the training log directory.
You can instead specify the path to another checkpoint by adding this to the evaluation command.
- `--debug`: print debug information and visualize environment.

## Acknowledgements

This work uses code from the following open-source projects and datasets:

#### CALVIN
Original:  [https://github.com/mees/calvin](https://github.com/mees/calvin)
License: [MIT](https://github.com/mees/calvin/blob/main/LICENSE)

#### Sentence-Transformers
Original:  [https://github.com/UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers)
License: [Apache 2.0](https://github.com/UKPLab/sentence-transformers/blob/master/LICENSE)

#### OpenAI CLIP
Original: [https://github.com/openai/CLIP](https://github.com/openai/CLIP)
License: [MIT](https://github.com/openai/CLIP/blob/main/LICENSE)


## License
MIT License
