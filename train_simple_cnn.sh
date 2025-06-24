#!/bin/bash

python train.py \
    --model-name SimpleCNN \
    --learning-rate 0.001 \
    --batch-size 64 \
    --epochs 15 \
    --model-save-dir saved_models_simple
