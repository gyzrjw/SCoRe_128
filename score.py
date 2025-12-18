# run_score_train.py
# 训练入口（不合并 Trainer），按要求映射 beta1_kl / beta2_kl，并在训练中不做验证

import argparse
import os
from functools import partial
from copy import deepcopy

import datasets
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForCausalLM
import torch
from peft import get_peft_model, TaskType, LoraConfig

from prompts import get_prompt_builder
from prompts.prompt_schemas import load_few_shot_prompts
from utils.generation_utils import load_config
from utils.eval_utils import RewardEvaluator

# IMPORTANT: 导入你已修改并保存好的 Trainer 模块（包含 SCoRETrainer 与 SCoREConfig）
# 如果你把 Trainer 文件命名为 score_trainer_fsdp.py，则以下导入保持不变。
# 若改名请相应修改下面的模块名。
from score_trainer_fsdp import SCoRETrainer, SCoREConfig


# ----------------------------
# 保留你提供的 collate 与 prompt helper（逐字保留）
# ----------------------------
def custom_collate_fn(features, config, collator):
    text_cols = [config['gold_col'], config['question_col']]
    batch_text = {}
    for col in text_cols:
        batch_text[col] = [f[col] for f in features]
    model_batch = collator([{'input_ids': f['input_ids']} for f in features])
    model_batch.update(batch_text)
    return model_batch


def add_input_ids(example, prompt_func):
    input_ids = prompt_func(example)
    example['input_ids'] = input_ids
    return example


# ----------------------------
# main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to your config (YAML/JSON) with data_path, model_path, question_col, etc.")
    args = parser.parse_args()

    # load config via your loader
    config = load_config(args.config_path)

    # create output dirs
    score_dir = os.path.join(config['cache_dir'], 'SCoRE')
    os.makedirs(score_dir, exist_ok=True)
    run_dir = os.path.join(score_dir, config['run_name'])
    os.makedirs(run_dir, exist_ok=True)

    # load dataset
    print(f"[INFO] Loading dataset from: {config['data_path']}")
    ds = datasets.load_from_disk(config['data_path'])

    # tokenizer
    print(f"[INFO] Loading tokenizer from: {config['model_path']}")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model_path'],
        use_fast=True,
        cache_dir=config['cache_dir']
    )

    # prompt builder & few-shot
    prompt_builder = get_prompt_builder(config['task_type'])
    initial_generation_few_shot = load_few_shot_prompts(
        config['few_shot_dir'],
        f"{config['task_type']}_initial"
    )

    # build initial-generation prompt function
    initial_generation_prompt_func = partial(
        prompt_builder.build_initial_generation_prompt,
        tokenizer=tokenizer,
        question_col=config['question_col'],
        few_shot_prompts=initial_generation_few_shot,
        tokenize=True
    )

    # add input_ids column to dataset
    ds = ds.map(partial(add_input_ids, prompt_func=initial_generation_prompt_func))

    # keep only necessary columns
    ds['train'] = ds['train'].select_columns([config['question_col'], config['gold_col'], 'input_ids'])
    # Do NOT include eval/test in trainer (user requested no validation during training)
    # keep ds['test'] if you need it externally but we won't pass it to trainer
    if 'test' in ds:
        ds['test'] = ds['test'].select_columns([config['question_col'], config['gold_col'], 'input_ids'])

    # reward function
    reward_function = RewardEvaluator(config)

    # load model + ref copy
    print(f"[INFO] Loading model from: {config['model_path']}")
    model = AutoModelForCausalLM.from_pretrained(
        config['model_path'],
        cache_dir=config['cache_dir'],
        torch_dtype=torch.bfloat16
    )
    ref_model = deepcopy(model)

    # apply LoRA if requested
    if config.get('use_lora', False):
        print(f"[INFO] Adding LoRA with R {config['lora_rank']}")
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=config['lora_rank'],
            lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'],
            target_modules=config.get('lora_target_modules', ['q_proj', 'k_proj', 'v_proj', 'o_proj']),
            init_lora_weights=True,
        )
        model = get_peft_model(model, lora_cfg)

    # wandb env (optional)
    os.environ["WANDB_PROJECT"] = config.get('wandb_project_name', os.environ.get("WANDB_PROJECT", "SCoRE"))
    os.environ["WANDB_DIR"] = config.get('cache_dir', os.environ.get("WANDB_DIR", "."))

    # map old config names to new SCoREConfig (beta1_kl / beta2_kl)
    score_config = SCoREConfig(
        output_dir=run_dir,
        exp_name=config['run_name'],
        seed=config['random_seed'],
        report_to=config.get('report_to', 'wandb'),
        per_device_train_batch_size=config['per_device_train_batch_size'],
        local_rollout_forward_batch_size=config['local_rollout_forward_batch_size'],
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
        response_length=config['max_tokens'],
        temperature=config['temperature'],
        total_episodes=config.get('total_episodes', None),
        num_sample_generations=0,
        beta1_kl=config.get('beta1_kl', 0.01),      # weak KL for Stage II
        beta2_kl=config.get('beta2_kl', 0.1),       # strong KL for Stage I (on y1)
        save_steps=config.get('save_steps', 1000),
        stage=config['train_stage'],
        stage2_alpha=config.get('stage2_alpha', 0.5),
        save_only_model=True,
    )

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # construct trainer (do NOT pass eval_dataset to trainer per your request)
    trainer = SCoRETrainer(
        config=score_config,
        algo_config=config,
        processing_class=tokenizer,
        policy=model,
        ref_policy=ref_model,
        reward_model=reward_function,
        train_dataset=ds['train'],
        data_collator=partial(custom_collate_fn, config=config, collator=DataCollatorWithPadding(tokenizer)),
        optimizers=(optimizer, None),
        prompt_builder=prompt_builder,
        # eval_dataset intentionally omitted
    )

    # start training
    trainer.train()


if __name__ == "__main__":
    main()
