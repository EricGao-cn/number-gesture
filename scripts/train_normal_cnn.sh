#!/bin/bash

uv run train.py \
    --model-name NormalCNN \
    --learning-rate 0.001 \
    --batch-size 64 \
    --epochs 15 \
    --model-save-dir saved_models/normal 
