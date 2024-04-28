# Self-supervised contrastive learning of echocardiogram videos enables label-efficient cardiac disease diagnosis

[**Gregory Holste**](https://gholste.me), Evangelos K. Oikonomou, Bobak J. Mortazavi, Zhangyang Wang, [**Rohan Khera**](https://www.cards-lab.org/team)

### [***arXiv preprint***](https://arxiv.org/abs/2207.11581) | 10 September 2023

(For a previous version of this paper presented at the ICML workshop, [IMLH 2022](https://sites.google.com/view/imlh2022/home?authuser=0), see the `imlh-2022/` directory.)

-----

## Description

<p align=center>
    <img src=figs/overview_fig_v4.png height=500>
</p>

**Overview of EchoCLR, a self-supervised learning approach for echocardiography.** Unlike standard contrastive learning methods, two distinct videos of each patient acquired during a single exam are randomly sampled and deemed positive pairs for “multi-instance” contrastive learning (A). The frames of each video are then randomly shuffled along the temporal axis and fed into a 3D CNN, which learns similar representations of distinct videos from the same patient and dissimilar representations of videos from different patients (B). These video-level representations are then used to directly predict the order of shuffled video frames. This frame reordering pretext task encourages temporal coherence, which we demonstrate to be beneficial for downstream echocardiogram video-based disease classification tasks (C). After self-supervised pretraining, the 3D CNN backbone can then be efficiently fine-tuned for cardiac disease classification based on very few labeled echocardiograms.

## Usage

To reproduce the results in the paper,
1. Prepare the conda environment.
- `conda env create -f echo.yml`
- `conda activate echo`
2. Run pretraining experiments. This will train the SimCLR, MI-SimCLR, and EchoCLR models.
- `bash ssl/run_ssl_experiments.sh`
3. Run fine-tuning experiments. For each init (Random, Kinetics-400, SimCLR, MI-SimCLR, and EchoCLR), this will fine-tune a model to predict severe aortic stenosis (AS) and left ventricular hypertrophy (LVH) on various ratios of the training data.
- `bash finetune/run_random_train_ratio_experiments.sh`
- `bash finetune/run_kinetics_train_ratio_experiments.sh`
- `bash finetune/run_simclr_train_ratio_experiments.sh`
- `bash finetune/run_mi-simclr_train_ratio_experiments.sh`
- `bash finetune/run_echoclr_train_ratio_experiments.sh`

## Citation

```
@misc{holste2023selfsupervised,
      title={Self-supervised contrastive learning of echocardiogram videos enables label-efficient cardiac disease diagnosis}, 
      author={Gregory Holste and Evangelos K. Oikonomou and Bobak J. Mortazavi and Zhangyang Wang and Rohan Khera},
      year={2023},
      eprint={2207.11581},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Contact

Contact me (Greg Holste) at gholste@utexas.edu with any questions!
