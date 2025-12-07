#!/bin/bash

BASE_MODEL="/root/autodl-tmp/data/math500/cache/SCoRE/test_lora/checkpoint-50_full"
ADAPTERS_PATH="/root/autodl-tmp/data/math500/cache/SCoRE/test_lora_stage2/"
CACHE_DIR="/root/autodl-tmp/data/v.moskvoretskii/cache/"

python merge_lora.py \
--base_model=$BASE_MODEL \
--adapters_path=$ADAPTERS_PATH \
--cache_dir=$CACHE_DIR
