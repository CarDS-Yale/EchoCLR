#!/usr/bin/bash

# SimCLR
python main.py \
    --data_dir /home/gih5/mounts/nfs_echo_yale/031522_echo_avs_preprocessed \
    --out_dir results \
    --model_name simclr \
    --batch_size 196 \
    --n_gpu 2 \
    --temperature 0.05 \
    --projection_dim 128 \
    --lr 0.1 \
    --num_epochs 520 \
    --clip_len 4 \
    --sampling_rate 1

# MI-SimCLR
python main.py \
    --data_dir /home/gih5/mounts/nfs_echo_yale/031522_echo_avs_preprocessed \
    --out_dir results \
    --model_name mi-simclr \
    --multi_instance \
    --batch_size 196 \
    --n_gpu 2 \
    --temperature 0.05 \
    --projection_dim 128 \
    --lr 0.1 \
    --num_epochs 300 \
    --clip_len 4 \
    --sampling_rate 1

# MI-SimCLR+FO
python main.py \
    --data_dir /home/gih5/mounts/nfs_echo_yale/031522_echo_avs_preprocessed \
    --out_dir results \
    --model_name mi-simclr-fo \
    --multi_instance \
    --frame_reordering \
    --batch_size 196 \
    --n_gpu 2 \
    --temperature 0.05 \
    --projection_dim 128 \
    --lr 0.1 \
    --num_epochs 300 \
    --clip_len 4 \
    --sampling_rate 1
