#  Language-Conditioned Robot Manipulation With Base Skill Priors Under Unstructured Data

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## [Paper](https://arxiv.org/pdf/2305.19075.pdf) | [Project Page](https://hk-zh.github.io/spil/)
![architecture](https://github.com/hk-zh/spil/assets/57254021/939dd916-f325-42a4-b02c-d1e8b0c1345a)

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
#### D -> D
- Pre-trained **SKILL Generator** can be downloaded [here](https://drive.google.com/drive/folders/1y4DM45ltB6mecrkjwF48d9NpJD0eYA1M?usp=sharing) (epoch-98)
- Pre-trained **Model** can be downloaded [here](https://drive.google.com/drive/folders/1CTcwDwhoSocZ5PdHROmOOqr3MXAM4thN?usp=sharing)
#### ABC -> D
- Pre-trained **SKILL Generator** can be downloaded [here](https://drive.google.com/drive/folders/1EbpG5zW4siQi5BJxXc2Js_4gxmTum2jA?usp=sharing) (epoch-86)
- Pre-trained **Model** can be downloaded [here](https://drive.google.com/drive/folders/1BDw8NXykYlsEyTAidVUqN1A-V-6VXUtV?usp=sharing)

## Hardware Requirements
Trained with:
- **GPU** - 1x NVIDIA Tesla V100 16GB
- **RAM** - 256GB
- **OS** - Ubuntu 20.04

## Training
Before you start your training, please remember to update the wandb account at
- conf/logger/wandb.yaml
- skill_generator/conf_sg/logger/wandb.yaml

To login your wandb account, first run:
```
wandb login --relogin
```

### SPIL model
To train the spil model with the maximum amount of available GPUS, run:
```
python spil/training.py trainer.gpus=-1 datamodule.root_data_dir=path/to/dataset model.action_decoder.sg_chk_path=path/to/skill_generator datamodule/datasets=vision_lang loss=your_loss_setting
```
To accelerate training process, the dataset can be first loaded into shared memory. (Note this way requires more RAM, please make sure your server has enough RAM)
```
python spil/training.py trainer.gpus=-1 datamodule.root_data_dir=path/to/dataset model.action_decoder.sg_chk_path=path/to/skill_generator datamodule/datasets=vision_lang_shm loss=your_loss_setting
```
- The `vision_lang_shm` option loads the CALVIN dataset into shared memory at the beginning of the training,
speeding up the data loading during training.
The preparation of the shared memory cache will take some time
(approx. 20 min at our SLURM cluster). 
- You can either use the following command to train the `skill-generator` or use a pre-trained one.

### Skill-Generator 
To train the skill generator, run:
```
python skill_generator/skill_generator/training.py trainer.gpus=-1 datamodule.root_data_dir=path/to/dataset 
```
Note that you should first train the skill-generator if you did not download the pre-trained skill generator.
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




https://github.com/hk-zh/spil/assets/57254021/6715986a-51c0-4159-9fcf-6bd644752c62



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
