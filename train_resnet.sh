#!/bin/bash

python train.py \
    --model-name Advanced_ResNet18 \
    --learning-rate 0.0005 \
    --batch-size 32 \
    --epochs 25 \
    --model-save-dir saved_models_resnet
