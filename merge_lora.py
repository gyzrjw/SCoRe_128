import torch
from pathlib import Path
from argparse import ArgumentParser
import os
import re
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from peft import PeftModel 
from transformers import AutoModelForCausalLM, AutoTokenizer



if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument("--base_model", help="full model", required=True, type=str)
    parser.add_argument("--adapters_path", help="path to adapters saving dir", required=True, type=str)
    parser.add_argument("--cache_dir", help="path to cache_dir", required=True, type=str)


    args = parser.parse_args()
    parent_directory = args.adapters_path
    prefix = 'checkpoint-'

    base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            #device_map='auto',
            torch_dtype=torch.bfloat16,
            cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            cache_dir=args.cache_dir
    )

    # ==================== START: MINIMAL CHANGE ====================

    # 1. 找到所有不带 "_full" 后缀的 checkpoint 目录
    checkpoint_dirs = [
        d for d in os.listdir(parent_directory)
        if os.path.isdir(os.path.join(parent_directory, d)) and d.startswith(prefix) and not d.endswith('_full')
    ]

    if not checkpoint_dirs:
        print(f"Error: No non-merged checkpoint directories found in '{parent_directory}'.")
        exit()

    # 2. 利用正则表达式从目录名中提取数字，并找到数字最大的那个目录
    latest_checkpoint_dir = max(checkpoint_dirs, key=lambda d: int(re.search(r'(\d+)', d).group(1)))
    
    print(f"Found latest checkpoint to merge: {latest_checkpoint_dir}")

    # 3. 将原来的循环体只对这一个最新的目录执行一次
    item = latest_checkpoint_dir # 将 item 设置为我们找到的最新目录
    item_path = os.path.join(parent_directory, item)

    # 检查目录是否真的存在 (安全检查)
    if os.path.isdir(item_path) and item.startswith(prefix):
        number_part = item[len(prefix):]

        model = PeftModel.from_pretrained(base_model, item_path) 
        merged_model = model.merge_and_unload() 

        result_path = os.path.join(parent_directory, f'checkpoint-{number_part}_full')
        os.makedirs(result_path, exist_ok=True)

        print(f"Saving merged model to {result_path}...")
        merged_model.save_pretrained(result_path, from_pt=True)
        tokenizer.save_pretrained(result_path)

        print(f"Successfully saved merged model to {result_path}")

        del model
        del merged_model
    
    # ===================== END: MINIMAL CHANGE =====================
