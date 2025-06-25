#!/bin/bash

uv run data/get_data.py

cd data || exit 1
mkdir -p dataset
mv 1/*/* dataset/ 2>/dev/null
rm -rf dataset/unknown
rm -rf 1

uv run data/data_info.py