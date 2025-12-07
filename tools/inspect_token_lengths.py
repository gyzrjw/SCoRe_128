#!/usr/bin/env python3
"""
Inspect token length distribution for prompts and references and suggest max_tokens.

Usage:
  python tools/inspect_token_lengths.py --config configs/score_config.yaml --sample 1000

The script will:
 - load the YAML config
 - load the dataset from `data_path` via `datasets.load_from_disk`
 - load tokenizer from `model_path` (use cache_dir)
 - build initial prompts using project's prompt builder (same as `score.py`)
 - compute token lengths for prompts and gold answers
 - print percentiles and suggest a `max_tokens` value based on model context length

"""
import argparse
import math
import os
import sys
from collections import Counter

import yaml
import numpy as np
import datasets
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from prompts import get_prompt_builder
from prompts.prompt_schemas import load_few_shot_prompts


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def percentile(arr, p):
    return int(np.percentile(arr, p))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/score_config.yaml')
    parser.add_argument('--sample', type=int, default=2000, help='max number of examples to sample')
    parser.add_argument('--show-top', type=int, default=10, help='show top n long examples')
    args = parser.parse_args()

    cfg = load_config(args.config)

    data_path = cfg.get('data_path')
    if data_path is None:
        print('config missing data_path', file=sys.stderr)
        sys.exit(1)

    print(f"Loading dataset from {data_path}...")
    # Some projects save as DatasetDict root, others as separate train/test dirs.
    d = None
    try:
        ds = datasets.load_from_disk(data_path)
        # decide which split to inspect: train by default
        split = 'train' if isinstance(ds, datasets.DatasetDict) and 'train' in ds else list(ds.keys())[0]
        print(f"Using split: {split}")
        d = ds[split] if isinstance(ds, datasets.DatasetDict) else ds
    except Exception:
        # fall back to train subdirectory
        train_path = os.path.join(data_path, 'train')
        print(f"Falling back to train split at {train_path} ...")
        d = datasets.load_from_disk(train_path)
        print(f"Loaded train split with {len(d)} examples")

    n = min(len(d), args.sample)
    print(f"Sampling {n} / {len(d)} examples")

    print(f"Loading tokenizer from {cfg.get('model_path')} (cache {cfg.get('cache_dir')})...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.get('model_path'), cache_dir=cfg.get('cache_dir'), use_fast=True)

    # build prompts
    prompt_builder = get_prompt_builder(cfg.get('task_type'))
    few_shot = load_few_shot_prompts(cfg.get('few_shot_dir'), f"{cfg.get('task_type')}_initial")

    def build_prompt(example):
        # prompt_builder usually expects tokenize=False to return text
        try:
            text = prompt_builder.build_initial_generation_prompt(
                sample=example,
                tokenizer=tokenizer,
                question_col=cfg.get('question_col'),
                few_shot_prompts=few_shot,
                tokenize=False,
            )
        except TypeError:
            # different signature: try positional
            text = prompt_builder.build_initial_generation_prompt(
                example, tokenizer, cfg.get('question_col'), few_shot, tokenize=False
            )
        # if builder returns list, join
        if isinstance(text, (list, tuple)):
            text = text[0]
        return text

    prompt_lens = []
    gold_lens = []
    combined_lens = []
    long_examples = []

    for i in range(n):
        ex = d[i]
        # build prompt text
        prompt_text = build_prompt(ex)
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        # gold/reference
        gold_col = cfg.get('gold_col')
        gold_text = ex.get(gold_col, '') if gold_col in ex else ''
        gold_ids = tokenizer.encode(str(gold_text), add_special_tokens=False) if gold_text is not None else []
        gold_len = len(gold_ids)

        prompt_lens.append(prompt_len)
        gold_lens.append(gold_len)
        combined_lens.append(prompt_len + gold_len)

        if prompt_len + gold_len > 0:
            long_examples.append((prompt_len + gold_len, i, prompt_len, gold_len, prompt_text, gold_text))

    def summarize(name, arr):
        arr = np.array(arr)
        print(f"\n{name} stats (n={len(arr)})")
        for p in (50, 75, 90, 95, 99, 100):
            print(f"  p{p}: {percentile(arr, p)}")
        print(f"  mean: {int(arr.mean())}  std: {int(arr.std())}")

    summarize('Prompt tokens', prompt_lens)
    summarize('Gold tokens', gold_lens)
    summarize('Prompt+Gold tokens', combined_lens)

    long_examples.sort(reverse=True)
    print(f"\nTop {args.show_top} longest prompt+gold examples:")
    for total, idx, p_len, g_len, p_text, g_text in long_examples[:args.show_top]:
        print(f"#{idx} total={total} prompt={p_len} gold={g_len}")
        print('  prompt snippet:', p_text[:200].replace('\n',' '))
        print('  gold snippet: ', str(g_text)[:200].replace('\n',' '))

    # model context length from tokenizer
    model_max = getattr(tokenizer, 'model_max_length', None)
    try:
        model_max = int(model_max)
    except Exception:
        model_max = None

    print('\nModel/tokenizer reported max length:', model_max)

    # Suggestion: choose generation max_tokens so that most prompts + generation fit within context
    med_prompt = int(np.median(prompt_lens))
    p95_prompt = percentile(prompt_lens, 95)
    p99_prompt = percentile(prompt_lens, 99)

    print('\nSuggested max_tokens (generation length) guidelines:')
    if model_max is None or model_max <= 0 or model_max > 1e6:
        print('  Could not determine model context length from tokenizer. Use an explicit safe value (e.g., 512 or 768).')
    else:
        for q, label in [(med_prompt, 'median prompt'), (p95_prompt, '95th pct prompt'), (p99_prompt, '99th pct prompt')]:
            safe_margin = 8
            gen_allowed = model_max - q - safe_margin
            gen_allowed = max(0, gen_allowed)
            print(f"  If using {label} ({q} tokens), max generation ≈ {gen_allowed} tokens (model_max {model_max} - prompt {q} - margin {safe_margin})")

    print('\nRecommendation:')
    print(' - If you expect long gold answers, set max_tokens to cover typical gold lengths (e.g., >= 95th percentile of gold tokens).')
    print(' - If prompt lengths vary a lot, choose generation length using 95th percentile prompt as worst-case, or increase model context (longer model) if needed.')
    print(' - Consider reducing few-shot examples or truncating prompts if prompt consumes too much context.')


if __name__ == '__main__':
    main()
