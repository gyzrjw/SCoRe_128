import argparse
import os
import yaml
import datasets
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from functools import partial

from prompts import get_prompt_builder
from prompts.prompt_schemas import load_few_shot_prompts
from utils.eval_utils import RewardEvaluator
from utils.generation_utils import load_config
from score_trainer_fsdp import SCoRETrainer
from copy import deepcopy
from trl import RLOOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
import torch
from dataclasses import dataclass, field
from peft import get_peft_model, TaskType, LoraConfig


@dataclass
#SCoREConfig 继承自 RLOOConfig，并扩展了字段。这些字段会被传入 SCoRETrainer，影响 reward 计算和 loss 配方
class SCoREConfig(RLOOConfig):
    stage: int = field(
        default=1,
        metadata={"help": "Stage of SCoRE Training"},
    )
    init_kl_coef: float = field(
        default=0.01,
        metadata={"help": "coef for initial generation KL"},
    )
    corr_kl_coef: float = field(
        default=0.01,
        metadata={"help": "coef for correction KL"},
    )
    stage2_alpha: float = field(
        default=0.5,
        metadata={"help": "bonus multiplier for correction improvement"},
    )
    offline_y1_ratio: float = field(
        default=0.3,
        metadata={"help": "Stage II: ratio of samples using offline base model y1 (防止分布漂移)"},
    )


#DataLoader 提供的原始列表（每条是一个字典，包含 input_ids, problem, answer 等）
#text_cols = [...]：指定需保留为纯文本列（在 score_config.yaml 中指定 gold_col, question_col）
# batch_text：循环收集每条 sample 的 question / gold 为列表，保留原始文本形式
#使用collator（DataCollatorWithPadding ）对 input_ids 进行 padding，生成 model_batch
#将 batch_text 中的文本列添加回 model_batch，形成最终批次数据
def custom_collate_fn(features, config, collator):
    # 1) Extract the text columns you want to keep
    #    but do NOT pass them to the default collator
    text_cols = [config['gold_col'], config['question_col']]
    batch_text = {}
    for col in text_cols:
        batch_text[col] = [f[col] for f in features]  # gather them as a list

    # 3) Use the default HF collator to pad the numeric parts only
    model_batch = collator([{'input_ids': f['input_ids']} for f in features])

    # 4) Attach the text lists back to the batch, as Python lists
    model_batch.update(batch_text)

    return model_batch



#prompt_func(example) 返回 tokenized prompt 的 input_ids
#将 prompt_func 生成的 input_ids 添加到 example 字典中
def add_input_ids(example, prompt_func):
    input_ids = prompt_func(example)
    example['input_ids'] = input_ids
    return example


def main():
    #接受 --config_path：指定 YAML 配置
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to your config (YAML/JSON) with data_path, model_path, question_col, etc.")
    args = parser.parse_args()

    # 1) 读取 config
    config = load_config(args.config_path)


    # 2) 创建 SCoRE 结果保存目录（将训练日志与 checkpoint 存放在cache_dir/SCoRE/run_name）
    score_dir = os.path.join(config['cache_dir'], 'SCoRE')
    os.makedirs(score_dir, exist_ok=True)
    run_dir = os.path.join(score_dir, config['run_name'])
    os.makedirs(run_dir, exist_ok=True)


    # 3) 加载数据集
    #dataset 在 data/... 已保存（HF datasets 保存格式），会返回 DatasetDict（train/test 等）
    #若 data_path 指向 .arrow 文件夹，会按 HF load 解析
    print(f"[INFO] Loading dataset from: {config['data_path']}")
    ds = datasets.load_from_disk(config['data_path'])

    # 4) 初始化 tokenizer
    #use_fast=True 优先用 Rust tokenizer（更快）
    #model_path 用于选择 tokenizer vocabulary / special tokens 等
    print(f"[INFO] Loading tokenizer from: {config['model_path']}")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model_path'],
        use_fast=True,
        cache_dir=config['cache_dir']

    )

    # 5) 加载 prompt builder 和 few-shot
    # get_prompt_builder 返回 ScoreMathPromptBuilder 或 QAPromptBuilder，根据 task_type（math/qa）
    # load_few_shot_prompts 从 few_shot_dir 加载 few-shot 示例
    # 这些 few-shot 示例会被插入到 prompt 中，提升模型生成质量
    # 
    prompt_builder = get_prompt_builder(config['task_type'])
    initial_generation_few_shot = load_few_shot_prompts(
        config['few_shot_dir'],
        f"{config['task_type']}_initial"
    )
    
    # 6) 构建 initial generation prompt function
    # partial 固定部分参数，返回新的函数 initial_generation_prompt_func
    initial_generation_prompt_func = partial(
        prompt_builder.build_initial_generation_prompt,
        tokenizer=tokenizer,
        question_col=config['question_col'],
        few_shot_prompts=initial_generation_few_shot,
        tokenize=True
    )

    # 7) 为数据集添加 input_ids 列
    # 使用 map 函数对数据集的每个 example 应用 add_input_ids
    # add_input_ids 使用 initial_generation_prompt_func 生成 input_ids
    ds = ds.map(partial(add_input_ids, prompt_func=initial_generation_prompt_func))

    # 8) 仅保留必要列，减少内存占用
    # 这里只保留 question_col, gold_col, input_ids 三列
    ds['train'] = ds['train'].select_columns([config['question_col'], config['gold_col'], 'input_ids'])
    ds['test'] = ds['test'].select_columns([config['question_col'], config['gold_col'], 'input_ids'])

    # 9) 初始化奖励函数
    reward_function = RewardEvaluator(config)

    # 10) 加载模型
    # 加载预训练 LLM；用 torch_dtype=bfloat16 减少内存（如果硬件支持）
    # ref_model 复制 base model，作为未更新的参考策略
    model = AutoModelForCausalLM.from_pretrained(
        config['model_path'],
        cache_dir=config['cache_dir'],
        torch_dtype=torch.bfloat16
    )
    ref_model = deepcopy(model)

    # 11) 应用 LoRA
    if config['use_lora']:
        print(f"[INFO] Adding LoRA with R {config['lora_rank']}")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=config['lora_rank'],
            lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'],
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            #config['lora_target_modules'],
            init_lora_weights=True,
            #use_dora=lora_args.dora
        )
        model = get_peft_model(model, lora_config)

    # 12) 设置wandb环境变量(项目名和缓存路径)
    os.environ["WANDB_PROJECT"] = config['wandb_project_name']
    os.environ["WANDB_DIR"] = config['cache_dir']
    os.environ["WANDB_CACHE_DIR"] = config['cache_dir']
    
    # 13) 配置 SCoRE 训练参数
    score_config = SCoREConfig(
        output_dir=run_dir,
        exp_name=config['run_name'],
        seed=config['random_seed'],
        report_to='wandb',
        per_device_train_batch_size=config['per_device_train_batch_size'],
        local_rollout_forward_batch_size=config['local_rollout_forward_batch_size'],    # 用于 batch_generation 的并行采样
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        response_length=config['max_tokens'],   # response_length 是 generation 的 max_new_tokens（init/corr)
        temperature=config['temperature'],
        total_episodes=config['total_episodes'],
        num_sample_generations=0,
        corr_kl_coef=config['corr_kl_coef'],
        init_kl_coef=config['init_kl_coef'],
        save_steps=config['save_steps'],
        stage=config['train_stage'],
        stage2_alpha=config['stage2_alpha'],
        offline_y1_ratio=config.get('offline_y1_ratio', 0.3),  # 默认 30%
        save_only_model=True    # 只保存模型权重，不保存优化器等状态
    )

    # 14) 初始化优化器
    # 若为 LoRA 模式，model.parameters() 返回包含 LoRA 参数（以及可能的 base 模型参数），
    # 但 get_peft_model 默认只把 LoRA 的参数设为需要梯度；确认 base 模型参数被冻结
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # 构造 Trainer 并传入各组件
    trainer = SCoRETrainer(
        config=score_config,
        algo_config=config,
        processing_class=tokenizer,
        policy=model, 
        ref_policy=ref_model, 
        reward_model=reward_function, 
        train_dataset=ds["train"],
        data_collator=partial(custom_collate_fn, config=config, collator=DataCollatorWithPadding(tokenizer)),
        optimizers=(optimizer, None),
        prompt_builder=prompt_builder,
    )
    
    # 15) 开始训练
    trainer.train()

if __name__ == "__main__":
    main()