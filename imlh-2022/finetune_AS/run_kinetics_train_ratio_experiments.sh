#!/usr/bin/bash

python main.py \
    --output_dir kinetics_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \

python main.py \
    --output_dir kinetics_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.75 \

python main.py \
    --output_dir kinetics_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.5 \

python main.py \
    --output_dir kinetics_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.25

python main.py \
    --output_dir kinetics_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.1 \

python main.py \
    --output_dir kinetics_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.1 \
    --lr 5e-5

python main.py \
    --output_dir kinetics_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.1 \
    --lr 1e-5

python main.py \
    --output_dir kinetics_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.05

python main.py \
    --output_dir kinetics_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.05 \
    --lr 5e-5

python main.py \
    --output_dir kinetics_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.05 \
    --lr 1e-5

python main.py \
    --output_dir kinetics_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.01

python main.py \
    --output_dir kinetics_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.01 \
    --lr 5e-5

python main.py \
    --output_dir kinetics_ft_results \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 88 \
    --label_smoothing 0.1 \
    --use_class_weights \
    --dropout_fc \
    --augment \
    --frac 0.01 \
    --lr 1e-5