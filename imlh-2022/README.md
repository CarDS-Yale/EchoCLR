# Self-Supervised Learning of Echocardiogram Videos Enables Data-Efficient Diagnosis of Severe Aortic Stenosis

Code repository for [Self-Supervised Learning of Echocardiogram Videos Enables Data-Efficient Diagnosis of Severe Aortic Stenosis](https://github.com/interpretable-ml-in-healthcare/IMLH2022/blob/main/63%5CCameraReady%5CEcho_AS_IMLH_2022_Camera_Ready.pdf), presented at [IMLH 2022](https://sites.google.com/view/imlh2022/home?authuser=0), an ICML workshop.

-----

## Description

<p align=center>
    <img src=figs/echo_avs_fig1_v3_imlh.png height=400>
</p>

Since acquiring high-quality labels for automated clinical diagnosis tasks is expensive, there is a need for *data-efficient* learning algorithms for automated diagnosis. We present a data-efficient approach for self-supervised learning (SSL) of echocardiogram videos, evaluating the quality of learned representations on the downstream task of severe aortic stenosis (AS) prediction. SSL is challenging for echocardiograms because (i) ultrasound images are very noisy and brittle to augmentation and (ii) most algorithms ignore the rich temporal content in echo videos. We tackle these challenges by (i) choosing *different* echo videos from the *same* patient as "positive pairs" for contrastive learning, and (ii) using an additional frame re-ordering pretext task, where we permute the frames of each video and train the network to predict the original order.

## Usage

To reproduce the results in the paper,
1. Prepare the conda environment.
- `conda env create -f echo.yml`
- `conda activate echo`
2. Run pretraining experiments. This will train the SimCLR, MI-SimCLR, and MI-SimCLR+FO models.
- `bash ssl/run_ssl_experiments.sh`
3. Run fine-tuning experiments. For each init (random, Kinetics-400, SimCLR, MI-SimCLR, and MI-SimCLR+FO), this will fine-tune a model to predict severe AS on various ratios of the training set.
- `bash finetune_AS/run_random_train_ratio_experiments.sh`
- `bash finetune_AS/run_kinetics_train_ratio_experiments.sh`
- `bash finetune_AS/run_simclr_train_ratio_experiments.sh`
- `bash finetune_AS/run_mi-simclr_train_ratio_experiments.sh`
- `bash finetune_AS/run_mi-simclr-fo_train_ratio_experiments.sh`

## Contact

Contact me (Greg Holste) at gholste@utexas.edu with any questions!