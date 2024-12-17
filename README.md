# MELA: Multi-event Localization Answering Framework for Video Question Answering [SAC2025]




# Code structure
```bash

# data & data preprocessing
./mela_data

# pretrained checkpoints
./mela_checkpoints


# MELA code
./lavis/models/mela_models/


# running scripts for MELA training
./run_scripts

```

# Setup

## Install Dependencies

1. (Optional) Creating conda environment

```bash
conda create -n mela python=3.8
conda activate mela
```

2. build from source

```bash
pip install -e .
```



# Dataset Preparation

We test our model on:
+ [STAR](https://star.csail.mit.edu/)

Please download original QA data and preprocess them via our [scripts](mela_data/).

# Download Pretrained Models
We provide [Q-former_loc weights](https://drive.google.com/drive/folders/1Gfu3BkVxhW_RgIOT15XBwdvktq1e4tKR?usp=drive_link).

# Download Detected Video Events using UniMD Event Detector
We provide [Detected Video Events annotation](https://drive.google.com/drive/folders/1-4wlf6IUO4pzetyWIGiOoCov9MWGEG8i?usp=drive_link).


# Training
We provide MELA training script examples as follows.

And please change your data path.

check files below.

     ./lavis/datasets/datasets/mc_video_vqa_datasets.py
     ./lavis/configs/datasets/star/defaults_qa.yaml
     ./lavis/models/mela_models/mela.py

## 1) Training Multi-event Localizer
```bash
sh run_scripts/mela/refinement/star_sr.sh
```

## 2) Extract Q-former_loc weights (change the model path first)


    mela_checkpoints/extract_qformer_loc.ipynb


## 3) Training Event-aware Answerer

Modify ./lavis/models/blip2_models/blip2.py
line 106 to your extracted qfrmer_loc weights path.

```bash
sh run_scripts/mela/finetune/star_ft.sh
```

# Acknowledgments
We thank the developers of [SeViLA](https://github.com/Yui010206/SeViLA),  [LAVIS](https://github.com/salesforce/LAVIS), [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2), [CLIP](https://github.com/openai/CLIP), [All-in-One](https://github.com/showlab/all-in-one), for their public code release.


# License

This project is licensed under the Apache-2.0 License.

