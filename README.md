# SPIL
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[<b>Language-Conditioned Robot Manipulation With Base Skill Priors Under Unstructured Data</b>](https://)

![SPIL](https://github.com/Hongkuan-Zhou/spil/assets/57254021/0e205350-4b50-4413-8792-b60d08c590d6)

## Installation
To begin, clone this repository locally
```bash
git clone --recurse-submodules https://github.com/Hongkuan-Zhou/spil
export ROOT=$(pwd)/spil

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
python spil/training.py trainer.gpus=-1 datamodule.root_data_dir=path/to/dataset datamodule/datasets=vision_lang_shm
```
The `vision_lang_shm` option loads the CALVIN dataset into shared memory at the beginning of the training,
speeding up the data loading during training.
The preparation of the shared memory cache will take some time
(approx. 20 min at our SLURM cluster). \
If you want to use the original data loader (e.g. for debugging) just override the command with `datamodule/datasets=vision_lang`. \
For an additional speed up, you can disable the evaluation callbacks during training by adding `~callbacks/rollout_lh`


### Ablations
Hierarchical Universal Language Conditioned Policies (HULC), (Oier et al. 2022)
```
python spil/training.py trainer.gpus=-1 datamodule.root_data_dir=path/to/dataset datamodule/datasets=vision_lang_shm model=mcil
datamodule=hulc loss=hulc
```

Multi-context imitation learning (MCIL), (Lynch et al., 2019):
```
python spil/training.py trainer.gpus=-1 datamodule.root_data_dir=path/to/dataset datamodule/datasets=vision_lang_shm model=mcil
datamodule=mcil
```

Goal-conditioned behavior cloning (GCBC), (Lynch et al., 2019):
```
python spil/training.py trainer.gpus=-1 datamodule.root_data_dir=path/to/dataset datamodule/datasets=vision_lang_shm model=gcbc
```


## Evaluation
See detailed inference instructions on the [CALVIN repo](https://github.com/mees/calvin#muscle-evaluation-the-calvin-challenge).
```
python spil/evaluation/evaluate_policy.py --dataset_path <PATH/TO/DATASET> --train_folder <PATH/TO/TRAINING/FOLDER>
```

Optional arguments:

- `--checkpoint <PATH/TO/CHECKPOINT>`: by default, the evaluation loads the last checkpoint in the training log directory.
You can instead specify the path to another checkpoint by adding this to the evaluation command.
- `--debug`: print debug information and visualize environment.
## Real-world Experiments


https://github.com/Hongkuan-Zhou/spil/assets/57254021/5e2df33f-7446-4113-8967-39109b7ffb09


## Acknowledgements

This work uses code from the following open-source projects and datasets:

#### HULC
Original: [https://github.com/lukashermann/hulc](https://github.com/lukashermann/hulc)
License: [MIT](https://github.com/mees/calvin/blob/main/LICENSE)

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