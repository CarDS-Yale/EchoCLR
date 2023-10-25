#!/usr/bin/bash

### SEVERE AS EXPERIMENTS ###
python main.py --label as \
    --output_dir mi-simclr_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --ssl /home/gih5/echo_ssl/pretrain_results/simclr_bs-196x2_clip-len-4_stride-1_tau-0.05_lr-0.1_simclr

python main.py --label as \
    --output_dir mi-simclr_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.75 \
    --ssl /home/gih5/echo_ssl/pretrain_results/simclr_bs-196x2_clip-len-4_stride-1_tau-0.05_lr-0.1_simclr

python main.py --label as \
    --output_dir mi-simclr_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.5 \
    --ssl /home/gih5/echo_ssl/pretrain_results/simclr_bs-196x2_clip-len-4_stride-1_tau-0.05_lr-0.1_simclr

python main.py --label as \
    --output_dir mi-simclr_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.25 \
    --ssl /home/gih5/echo_ssl/pretrain_results/simclr_bs-196x2_clip-len-4_stride-1_tau-0.05_lr-0.1_simclr

python main.py --label as \
    --output_dir mi-simclr_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.1 \
    --ssl /home/gih5/echo_ssl/pretrain_results/simclr_bs-196x2_clip-len-4_stride-1_tau-0.05_lr-0.1_simclr

python main.py --label as \
    --output_dir mi-simclr_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.1 \
    --lr 5e-4 \
    --ssl /home/gih5/echo_ssl/pretrain_results/simclr_bs-196x2_clip-len-4_stride-1_tau-0.05_lr-0.1_simclr

python main.py --label as \
    --output_dir mi-simclr_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.1 \
    --lr 1e-5 \
    --ssl /home/gih5/echo_ssl/pretrain_results/simclr_bs-196x2_clip-len-4_stride-1_tau-0.05_lr-0.1_simclr

python main.py --label as \
    --output_dir mi-simclr_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.05 \
    --ssl /home/gih5/echo_ssl/pretrain_results/simclr_bs-196x2_clip-len-4_stride-1_tau-0.05_lr-0.1_simclr

python main.py --label as \
    --output_dir mi-simclr_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.05 \
    --lr 5e-4 \
    --ssl /home/gih5/echo_ssl/pretrain_results/simclr_bs-196x2_clip-len-4_stride-1_tau-0.05_lr-0.1_simclr

python main.py --label as \
    --output_dir mi-simclr_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.05 \
    --lr 1e-5 \
    --ssl /home/gih5/echo_ssl/pretrain_results/simclr_bs-196x2_clip-len-4_stride-1_tau-0.05_lr-0.1_simclr

python main.py --label as \
    --output_dir mi-simclr_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.01 \
    --ssl /home/gih5/echo_ssl/pretrain_results/simclr_bs-196x2_clip-len-4_stride-1_tau-0.05_lr-0.1_simclr

python main.py --label as \
    --output_dir mi-simclr_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.01 \
    --lr 5e-4 \
    --ssl /home/gih5/echo_ssl/pretrain_results/simclr_bs-196x2_clip-len-4_stride-1_tau-0.05_lr-0.1_simclr

python main.py --label as \
    --output_dir mi-simclr_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.01 \
    --lr 1e-5 \
    --ssl /home/gih5/echo_ssl/pretrain_results/simclr_bs-196x2_clip-len-4_stride-1_tau-0.05_lr-0.1_simclr
### END SEVERE AS EXPERIMENTS ###

### LVH EXPERIMENTS ###
python main.py --label lvh \
    --output_dir mi-simclr_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --ssl /home/gih5/echo_ssl/pretrain_results/simclr_bs-196x2_clip-len-4_stride-1_tau-0.05_lr-0.1_simclr

python main.py --label lvh \
    --output_dir mi-simclr_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.75 \
    --ssl /home/gih5/echo_ssl/pretrain_results/simclr_bs-196x2_clip-len-4_stride-1_tau-0.05_lr-0.1_simclr

python main.py --label lvh \
    --output_dir mi-simclr_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.5 \
    --ssl /home/gih5/echo_ssl/pretrain_results/simclr_bs-196x2_clip-len-4_stride-1_tau-0.05_lr-0.1_simclr

python main.py --label lvh \
    --output_dir mi-simclr_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.25 \
    --ssl /home/gih5/echo_ssl/pretrain_results/simclr_bs-196x2_clip-len-4_stride-1_tau-0.05_lr-0.1_simclr

python main.py --label lvh \
    --output_dir mi-simclr_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.1 \
    --ssl /home/gih5/echo_ssl/pretrain_results/simclr_bs-196x2_clip-len-4_stride-1_tau-0.05_lr-0.1_simclr

python main.py --label lvh \
    --output_dir mi-simclr_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.1 \
    --lr 5e-4 \
    --ssl /home/gih5/echo_ssl/pretrain_results/simclr_bs-196x2_clip-len-4_stride-1_tau-0.05_lr-0.1_simclr

python main.py --label lvh \
    --output_dir mi-simclr_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.1 \
    --lr 1e-5 \
    --ssl /home/gih5/echo_ssl/pretrain_results/simclr_bs-196x2_clip-len-4_stride-1_tau-0.05_lr-0.1_simclr

python main.py --label lvh \
    --output_dir mi-simclr_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.05 \
    --ssl /home/gih5/echo_ssl/pretrain_results/simclr_bs-196x2_clip-len-4_stride-1_tau-0.05_lr-0.1_simclr

python main.py --label lvh \
    --output_dir mi-simclr_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.05 \
    --lr 5e-4 \
    --ssl /home/gih5/echo_ssl/pretrain_results/simclr_bs-196x2_clip-len-4_stride-1_tau-0.05_lr-0.1_simclr

python main.py --label lvh \
    --output_dir mi-simclr_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.05 \
    --lr 1e-5 \
    --ssl /home/gih5/echo_ssl/pretrain_results/simclr_bs-196x2_clip-len-4_stride-1_tau-0.05_lr-0.1_simclr

python main.py --label lvh \
    --output_dir mi-simclr_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.01 \
    --ssl /home/gih5/echo_ssl/pretrain_results/simclr_bs-196x2_clip-len-4_stride-1_tau-0.05_lr-0.1_simclr

python main.py --label lvh \
    --output_dir mi-simclr_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.01 \
    --lr 5e-4 \
    --ssl /home/gih5/echo_ssl/pretrain_results/simclr_bs-196x2_clip-len-4_stride-1_tau-0.05_lr-0.1_simclr

python main.py --label lvh \
    --output_dir mi-simclr_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.01 \
    --lr 1e-5 \
    --ssl /home/gih5/echo_ssl/pretrain_results/simclr_bs-196x2_clip-len-4_stride-1_tau-0.05_lr-0.1_simclr
### END LVH EXPERIMENTS ###