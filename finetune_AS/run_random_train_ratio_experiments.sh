#!/usr/bin/bash

python main.py \
    --output_dir random_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --rand_init

python main.py \
    --output_dir random_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.75 \
    --rand_init

python main.py \
    --output_dir random_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.5 \
    --rand_init

python main.py \
    --output_dir random_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.25 \
    --rand_init

python main.py \
    --output_dir random_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.1 \
    --rand_init

python main.py \
    --output_dir random_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.1 \
    --lr 5e-5 \
    --rand_init

python main.py \
    --output_dir random_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.1 \
    --lr 1e-5 \
    --rand_init

python main.py \
    --output_dir random_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.05 \
    --rand_init

python main.py \
    --output_dir random_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.05 \
    --lr 5e-5 \
    --rand_init

python main.py \
    --output_dir random_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.05 \
    --lr 1e-5 \
    --rand_init

python main.py \
    --output_dir random_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.01 \
    --rand_init

python main.py \
    --output_dir random_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.01 \
    --lr 5e-5 \
    --rand_init

python main.py \
    --output_dir random_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.01 \
    --lr 1e-5 \
    --rand_init
