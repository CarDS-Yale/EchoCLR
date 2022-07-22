# Self-Supervised Learning of Echocardiogram Videos Enables Data-Efficient Diagnosis of Severe Aortic Stenosis

**[WORK IN PROGRESS: PREPROCESSING + ANALYSIS/VISUALIZATION CODE COMING SOON...]**

Code repository for [Self-Supervised Learning of Echocardiogram Videos Enables Data-Efficient Diagnosis of Severe Aortic Stenosis](https://github.com/interpretable-ml-in-healthcare/IMLH2022/blob/main/63%5CCameraReady%5CEcho_AS_IMLH_2022_Camera_Ready.pdf), presented at [IMLH 2022](https://sites.google.com/view/imlh2022/home?authuser=0), an ICML workshop.

-----

<p align=center>
    <img src=figs/echo_avs_fig1_v3_imlh.png height=400>
</p>

Since acquiring high-quality labels for automated clinical diagnosis tasks is very expensive, there is a need for *data-efficient* learning algorithms for automated diagnosis. We present a data-efficient approach for self-supervised learning (SSL) of echocardiogram videos, evaluating the quality of learned representations with the downstream task of severe aortic stenosis (AS) prediction. SSL is challenging for echocardiograms because (i) ultrasound images are very noisy and brittle to augmentation and (ii) most algorithms ignore the rich temporal content in echo videos. We tackle these challenges by (i) choosing *different* echo videos from the *same* patient as "positive pairs" for contrastive learning, and (ii) use an additional frame re-ordering pretext task, whereby we permute the frames of a video and train the network to predict the original order.

## Workflow

