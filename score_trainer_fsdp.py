import gc
import math
import os
import time
from collections import defaultdict
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    BaseImageProcessor,
    DataCollatorWithPadding,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    TrainerControl,
    get_cosine_schedule_with_warmup
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback

# Assume these come from TRL’s code (as in RLOOTrainer):
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import (
    OnlineTrainerState,
    batch_generation,
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    get_reward,
    prepare_deepspeed,
    print_rich_table,
    selective_log_softmax,
    truncate_response,
)
from trl.trainer.rloo_config import RLOOConfig  # or create a new SCoREConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, log_table_to_comet_experiment
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from copy import deepcopy



# TODO: Add Scheduler, add validation

INVALID_LOGPROB = 1.0

class SCoRETrainer(Trainer):
    """
    A single-stage SCoRE algorithm implemented in the style of TRL's RLOOTrainer.

    先生成 initial answer (y1) — 并计算与 ref policy 的 KL 约束
    再生成 correction (y2) — 计算 reward (r2)
    最终目标：R = r2 - beta * KL(y1)（Stage 1） 或 Stage 2 的更复杂配方
    使用 REINFORCE (policy gradient) 对 y2 的 token log-prob 做梯度更新（并在 Stage2 中同时更新 y1）

    Key differences from standard PPO/POLICY GRAD approaches:
      - We generate an INITIAL answer (which gets a KL penalty vs. ref policy).
      - Then we generate a CORRECTION from the policy, which gets a reward from
        the reward/cost function.
      - Final scalar = Reward(correction) - beta * KL(initial).
      - Use REINFORCE on the correction tokens to update the policy.

    Everything else (Accelerator, logging, etc.) is kept consistent
    with the RLOOTrainer style.
    """
    _tag_names = ["trl", "score"]  # for optional tracking in your model config

    def __init__(
        self,
        config: RLOOConfig,
        algo_config,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ],
        policy: nn.Module,
        ref_policy: nn.Module,
        prompt_builder,
        reward_model: Union[nn.Module, Callable[[list[str]], list[float]]],
        train_dataset: Dataset,
        data_collator: Optional[DataCollatorWithPadding] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler] = (None, None),
        callbacks: Optional[list[TrainerCallback]] = None,
    ) -> None:
        """
        Very similar to RLOOTrainer.__init__, except we note that we only do single-step REINFORCE
        logic inside the train loop.
        """
        #保证ref_policy与 policy 不是同一对象
        if ref_policy is policy:    
            raise ValueError(
                "`policy` and `ref_policy` cannot be the same object. If you want `ref_policy` to be the "
                "same as `policy`, pass a *copy* or pass `None` if using PEFT's read-only approach."
            )

        # 显式冻结 ref_policy 的所有参数，确保不会被更新
        for param in ref_policy.parameters():
            param.requires_grad = False
        ref_policy.eval()  # 设置为评估模式

        self.args = config  # For TRL, a config derived from RLOOConfig or similar
        args = config
        self.algo_config = algo_config
        self.processing_class = processing_class
        self.policy = policy
        self.ref_policy = ref_policy
        self.reward_model = reward_model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = None  # used by HF Trainer if re-creating optimizers
        self.prompt_builder = prompt_builder

        
        # 如果 data_collator 未给定，则基于 tokenizer 创建默认 collator；
        # collator 用于数据打包（padding/attention mask）
        if data_collator is None:
            data_collator = DataCollatorWithPadding(self.processing_class)
        self.data_collator = data_collator

        # 禁用 dropout
        for module in [policy, ref_policy, reward_model]:
            if isinstance(module, nn.Module):
                disable_dropout_in_model(module)

        # GenerationConfig 设置（initial / correction）
        # 两者通常相同，可根据需要调整
        self.init_generation_config = GenerationConfig(
            max_new_tokens=args.response_length,  # or some separate param, e.g. args.initial_answer_length
            temperature=args.temperature,
            #top_k=0,
            top_p=0.8,
            do_sample=True,
            pad_token_id=None,
            eos_token_id=None,
        )
        self.corr_generation_config = GenerationConfig(
            max_new_tokens=args.response_length,  # or separate param, e.g. correction_length
            temperature=args.temperature,
            #top_k=0,
            top_p=0.8,
            do_sample=True,
            pad_token_id=None,
            eos_token_id=None,
        )

        # Construct the dataloader
        self.train_dataset_len = len(train_dataset)
        if args.total_episodes is None:  # allow user to define episodes in terms of epochs
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)

        # Build accelerator
        accelerator = Accelerator()
        self.accelerator = accelerator
        self.accelerator.state.deepspeed_plugin.deepspeed_config['gradient_accumulation_steps'] = self.algo_config['gradient_accumulation_steps']
        args.world_size = accelerator.num_processes

        # This part is from RLOO: computing local_batch_size, micro_batch_size, etc.
        args.local_batch_size = (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
        )
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        # we do not do multiple mini-batches in this example, so skip that part

        # total number of train steps
        args.num_total_batches = math.ceil(args.total_episodes / args.batch_size)
        # name runs etc.
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(time_tensor, 0).item()
        args.run_name = f"{args.exp_name}"

        # Seeds, directories, etc.
        self.local_seed = args.seed
        torch.manual_seed(args.seed)

        # Prepare data loader
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=args.local_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True,
        )
        self.model = policy  # HF Trainer expects self.model
        self.model, self.optimizer, self.dataloader = accelerator.prepare(self.model, self.optimizer, self.dataloader)
        self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=self.algo_config['num_warmup_steps'], num_training_steps=args.num_total_batches)


        # reset local seed
        torch.manual_seed(self.local_seed)

        # Prepare eval dataloader if needed
        if self.eval_dataset is not None:
            self.eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=args.per_device_eval_batch_size,
                collate_fn=self.data_collator,
                drop_last=True,
            )
            self.eval_dataloader = accelerator.prepare(self.eval_dataloader)
        else:
            self.eval_dataloader = None

        # If using DeepSpeed / FSDP
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        if self.is_deepspeed_enabled:
            if isinstance(self.reward_model, nn.Module):
                self.reward_model = prepare_deepspeed(
                    self.reward_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )
            self.ref_policy = prepare_deepspeed(
                self.ref_policy, args.per_device_train_batch_size, args.fp16, args.bf16
            )
            self.deepspeed = self.model
        else:
            self.ref_policy = self.ref_policy.to(self.accelerator.device)
            if isinstance(self.reward_model, nn.Module):
                self.reward_model = self.reward_model.to(self.accelerator.device)

        # Create optimizer if not passed in
        if self.optimizer is None:
            self.create_optimizer_and_scheduler(num_training_steps=args.num_total_batches)

        # Setup HF Trainer state + callbacks
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)


        self.control = TrainerControl()
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
            save_steps=args.save_steps
        )
        self.current_flos = 0
        self.hp_search_backend = None

        # Create local dir, push to hub, etc.
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        # Tag model if needed
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

    def train(self):
        """
        Single-stage SCoRE training loop:
          1) Generate initial answer (and compute KL vs. ref policy).
          2) Generate correction (and get reward from reward_model).
          3) Final reward = reward(correction) - beta * KL(initial).
          4) REINFORCE update on the correction’s log-probs.
        """
        args = self.args
        accelerator = self.accelerator
        device = accelerator.device
        dataloader = self.dataloader

        # internal trainer states
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches  # or something else
        self.state.num_train_epochs = args.num_train_epochs  # for logging

        # Start
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
        start_time = time.time()

        # Reusable function to get next batch, infinitely
        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())
        self.model.train()

        # ================= [新增代码开始] =================
        # 手动指定要恢复的 checkpoint 路径和步数
        # resume_step = 300  # 你保存的 checkpoint 步数
        # resume_path = "/root/autodl-tmp/data/math500/cache/SCoRE/score_stage2/checkpoint-300" # 请确认路径正确
        
        # if resume_step > 0:
        #     print(f"🔥 Resuming training from step {resume_step}...")
            
        #     # 1. 加载 Policy 权重 (注意：只加载给 self.model，不要动 self.ref_policy)
        #     # 你的代码使用了 Accelerator，所以最好通过 unwrap 加载，或者直接加载
        #     # 这里假设保存的是 safetensors 或 bin，且是 LoRA 适配器
        #     from peft import PeftModel
        #     # 重新加载适配器权重
        #     self.model.load_adapter(resume_path, adapter_name="default")
            
        #     # 2. 恢复状态变量
        #     self.state.global_step = resume_step
        #     self.state.episode = resume_step * args.batch_size
            
        #     # 注意：优化器状态(Optimizer State)在这里很难完美恢复，
        #     # 因为你的代码是在 train 内部创建的 optimizer。
        #     # 接受“学习率重新预热”通常是可以接受的。
        # ================= [新增代码结束] =================

        ema_stats = {} 
        ema_alpha = 0.05  # 平滑系数

        self.processing_class.padding_side = "left"
        if self.processing_class.pad_token_id is None:
            self.processing_class.pad_token_id = self.processing_class.eos_token_id

        for step_idx in range(1, args.num_total_batches + 1):
            data = next(iter_dataloader)
            self.state.episode += args.batch_size

            # ------------------------
            # 1) Generate INITIAL and CORRECTION (采样策略按阶段区分)
            # ------------------------
            with torch.no_grad():
                queries = data["input_ids"].to(device).long()

                with unwrap_model_for_generation(
                    self.model, accelerator, gather_deepspeed3_params=args.ds3_gather_for_generation
                ) as unwrapped_model:
                    
                    # ======== 关键修复：采样策略按训练阶段区分 ========
                    if args.stage == 1:
                        # Stage I: 100% 使用 ref_policy (base model)
                        # 目标：学习如何纠正 base model 的次优答案
                        init_outputs, _ = batch_generation(
                            self.ref_policy,  # ✓ 修复：Stage I 必须用 ref_policy
                            queries,
                            args.local_rollout_forward_batch_size,
                            self.processing_class.pad_token_id,
                            self.init_generation_config,
                        )
                    else:
                        # Stage II: 样本级混合采样（论文 Section 5.3）
                        # 对每个样本独立决定使用 ref_policy 或当前策略
                        # 这样确保梯度更新的稳定性
                        batch_size = queries.shape[0]
                        use_offline_mask = torch.rand(batch_size, device=device) < args.offline_y1_ratio
                        
                        # 分别对两组样本生成
                        offline_indices = torch.where(use_offline_mask)[0]
                        online_indices = torch.where(~use_offline_mask)[0]
                        
                        outputs_list = []
                        indices_list = []

                        if len(offline_indices) > 0:
                            offline_queries = queries[offline_indices]
                            offline_outputs, _ = batch_generation(
                                self.ref_policy,
                                offline_queries,
                                args.local_rollout_forward_batch_size,
                                self.processing_class.pad_token_id,
                                self.init_generation_config,
                            )
                            outputs_list.append(offline_outputs)
                            indices_list.append(offline_indices)
                        
                        if len(online_indices) > 0:
                            online_queries = queries[online_indices]
                            online_outputs, _ = batch_generation(
                                unwrapped_model,
                                online_queries,
                                args.local_rollout_forward_batch_size,
                                self.processing_class.pad_token_id,
                                self.init_generation_config,
                            )
                            outputs_list.append(online_outputs)
                            indices_list.append(online_indices)
                        
                        # Pad to same length before concatenation
                        if len(outputs_list) > 1:
                            max_len = max(x.shape[1] for x in outputs_list)
                            for i in range(len(outputs_list)):
                                if outputs_list[i].shape[1] < max_len:
                                    outputs_list[i] = torch.nn.functional.pad(
                                        outputs_list[i], 
                                        (0, max_len - outputs_list[i].shape[1]), 
                                        value=self.processing_class.pad_token_id
                                    )
                        
                        init_outputs = torch.cat(outputs_list, dim=0)
                        combined_indices = torch.cat(indices_list, dim=0)
                        sort_order = torch.argsort(combined_indices)
                        init_outputs = init_outputs[sort_order]

                    init_context_len = queries.shape[1]
                    init_answers = init_outputs[:, init_context_len:]

                    # 生成 correction 的输入
                    init_answer_texts = self.processing_class.batch_decode(init_answers, skip_special_tokens=True)
                    corr_inputs = build_correction_inputs_for_batch(
                        data, 
                        init_answer_texts,
                        self.processing_class,
                        self.prompt_builder,
                        question_col=self.algo_config['question_col'],
                    ).to(device)
                
                    # y2 (correction) 始终从当前训练的策略采样
                    corr_outputs, _ = batch_generation(
                        unwrapped_model,  # Stage I/II 都从 self.model 采样 y2
                        corr_inputs,
                        args.local_rollout_forward_batch_size,
                        self.processing_class.pad_token_id,
                        self.init_generation_config,
                    )
                
                    corr_context_len = corr_inputs.shape[1]
                    corr_tokens = corr_outputs[:, corr_context_len:]

                # 计算 reference policy 的 log prob (用于 KL，不需要梯度)
                ref_init_out = forward(self.ref_policy, init_outputs, self.processing_class.pad_token_id)
                ref_init_logits = ref_init_out.logits[:, init_context_len - 1 : -1]
                # ref_init_logits /= args.temperature + 1e-7
                ref_init_logprob = selective_log_softmax(ref_init_logits, init_answers)
                mask_init = (init_answers == self.processing_class.pad_token_id)
                ref_init_logprob = ref_init_logprob.masked_fill_(mask_init, 0)

                ref_corr_out = forward(self.ref_policy, corr_outputs, self.processing_class.pad_token_id)
                ref_corr_logits = ref_corr_out.logits[:, corr_context_len - 1 : -1]
                # ref_corr_logits /= args.temperature + 1e-7
                ref_corr_logprob = selective_log_softmax(ref_corr_logits, corr_tokens)
                mask_corr = (corr_tokens == self.processing_class.pad_token_id)
                ref_corr_logprob = ref_corr_logprob.masked_fill_(mask_corr, 0)

                del ref_init_out, ref_init_logits, ref_corr_out, ref_corr_logits
                torch.cuda.empty_cache()

            # 计算 reward
            corr_output_text = self.processing_class.batch_decode(corr_tokens, skip_special_tokens=True)
            with torch.no_grad():
                reward_corr = torch.tensor([
                    self.reward_model(model_answer=corr_output, ground_truth=reference)
                    for (corr_output, reference) in zip(corr_output_text, data[self.algo_config["gold_col"]])
                ], dtype=torch.float, device=device)

                reward_init = torch.tensor([
                    self.reward_model(model_answer=init_output, ground_truth=reference)
                    for (init_output, reference) in zip(init_answer_texts, data[self.algo_config["gold_col"]])
                ], dtype=torch.float, device=device)

            torch.cuda.empty_cache()
            gc.collect()

            # ------------------------
            # 3) 计算 Loss (符合论文公式)
            # ------------------------
            # 论文 Stage I: max E[r(y2)] - β·KL(π_θ(·|x1) || π_ref(·|x1))
            # 论文 Stage II: max E[r(y1) + r̃(y2)] - β·(KL_1 + KL_2)
            # 
            # 关键点：KL 是独立的正则化项，需要有梯度！
            # Policy gradient 部分使用 baseline 减均值

            micro_step = 0
            total_loss = 0
            # 用于日志记录 KL，Stage 1/2 均累积两项
            total_kl_init = 0
            total_kl_corr = 0
            
            for micro_start in range(0, args.local_batch_size, args.per_device_train_batch_size):                
                micro_end = micro_start + args.per_device_train_batch_size
                mb_idx = slice(micro_start, micro_end)
                mb_corr_outputs = corr_outputs[mb_idx]
                mb_corr_tokens = corr_tokens[mb_idx]
                mb_init_tokens = init_answers[mb_idx]
                mb_init_outputs = init_outputs[mb_idx]
                mb_reward_corr = reward_corr[mb_idx]
                mb_reward_init = reward_init[mb_idx]
                mb_ref_init_logprob = ref_init_logprob[mb_idx]
                mb_ref_corr_logprob = ref_corr_logprob[mb_idx]
                mb_mask_init = mask_init[mb_idx]
                mb_mask_corr = mask_corr[mb_idx]
                mb_queries = queries[mb_idx]
                mb_init_context_len = mb_queries.shape[1]

                # ========== 计算当前策略的 log prob (需要梯度) ==========
                # Stage I 和 Stage II 都需要计算 y1 和 y2 的 logprob
                # 区别：
                # - Stage I: y1 的 logprob 仅用于 KL 约束，不做 PG
                # - Stage II: y1 的 logprob 用于 KL 约束 + PG
                
                # y1 的 log prob（Stage I/II 都需要，用于 KL 约束）
                out_init = forward(self.model, mb_init_outputs, self.processing_class.pad_token_id)
                logits_init = out_init.logits[:, mb_init_context_len - 1 : -1]
                logprob_init = selective_log_softmax(logits_init, mb_init_tokens)
                logprob_init = logprob_init.masked_fill_(mb_mask_init, 0)
                
                # y2 的 log prob（Stage I/II 都需要）
                out_corr = forward(self.model, mb_corr_outputs, self.processing_class.pad_token_id)
                logits_corr = out_corr.logits[:, corr_context_len - 1 : -1]
                logprob_corr = selective_log_softmax(logits_corr, mb_corr_tokens)
                logprob_corr = logprob_corr.masked_fill_(mb_mask_corr, 0)

                # ========== 计算 Policy Gradient Loss ==========
                sum_lp_corr = logprob_corr.sum(dim=1)

                if args.stage == 1:
                    # ============================================================
                    # Stage I: 初始化阶段（论文 Section 4.1 + 5.1）
                    # ============================================================
                    # 完整目标: max E[r(y2)] - β1·D_KL(π_θ(y1|x) || π_ref(y1|x)) 
                    #                          - β2·D_KL(π_θ(y2|x,y1) || π_ref(y2|x,y1))
                    # 
                    # 关键点：
                    # 1. y1 来自 ref_policy（离线数据），不做 Policy Gradient
                    # 2. 但需要 KL 约束，确保模型在生成 y1 时保持接近 base model
                    # 3. y2 做 Policy Gradient + KL 约束，学习纠错能力
                    # 4. 双重 KL 约束防止模型崩溃到"直接解决方案"
                    # ============================================================
                    
                    with torch.no_grad():
                        baseline_corr = reward_corr.mean()
                        advantage_corr = (mb_reward_corr - baseline_corr)

                    # Policy Gradient: 只优化 y2
                    sum_lp_corr = logprob_corr.sum(dim=1)
                    loss_pg_corr = -(advantage_corr * sum_lp_corr).mean()

                    # KL 正则化: 约束 y1 和 y2
                    valid_mask_init = (~mb_mask_init).float()
                    valid_mask_corr = (~mb_mask_corr).float()
                    
                    # y1 的 KL: 确保模型生成 y1 时接近 base model
                    kl_init_per_token = (logprob_init - mb_ref_init_logprob) * valid_mask_init
                    len_init = valid_mask_init.sum(dim=1).clamp(min=1.0)
                    kl_init_per_sample = kl_init_per_token.sum(dim=1) / len_init
                    kl_init_per_sample = torch.clamp(kl_init_per_sample, min=0.0)
                    loss_kl_init = args.init_kl_coef * kl_init_per_sample.mean()
                    
                    # y2 的 KL: 确保纠错时不过度偏离
                    kl_corr_per_token = (logprob_corr - mb_ref_corr_logprob) * valid_mask_corr
                    len_corr = valid_mask_corr.sum(dim=1).clamp(min=1.0)
                    kl_corr_per_sample = kl_corr_per_token.sum(dim=1) / len_corr
                    kl_corr_per_sample = torch.clamp(kl_corr_per_sample, min=0.0)
                    loss_kl_corr = args.corr_kl_coef * kl_corr_per_sample.mean()

                    # Stage I 总损失: PG(y2) + KL(y1) + KL(y2)
                    loss = loss_pg_corr + loss_kl_init + loss_kl_corr
                    
                    # 日志记录
                    total_kl_init += kl_init_per_sample.detach().mean().item()
                    total_kl_corr += kl_corr_per_sample.detach().mean().item()

                else:
                    # ============================================================
                    # Stage II: 联合优化阶段（论文 Section 4.2）
                    # ============================================================
                    # 论文公式: max E[r(y1) + r̃(y2)] - β·(KL_1 + KL_2)
                    # 其中 r̃(y2) = r(y2) + α·(r(y2) - r(y1))
                    # 
                    # 关键理解：
                    # "联合优化"是指整体目标函数包含两项奖励
                    # 但 REINFORCE 梯度应该分离：
                    # - y1 的梯度由 r(y1) 驱动
                    # - y2 的梯度由 r̃(y2) 驱动
                    # ============================================================
                    sum_lp_init = logprob_init.sum(dim=1)
                    
                    # 计算 y2 的增强奖励（带 reward bonus）
                    r2_tilde = mb_reward_corr + args.stage2_alpha * (mb_reward_corr - mb_reward_init)
                    
                    with torch.no_grad():
                        # 使用整个 local batch 计算 baseline（减少方差）
                        baseline_init = reward_init.mean()
                        
                        all_r2_tilde = reward_corr + args.stage2_alpha * (reward_corr - reward_init)
                        baseline_corr = all_r2_tilde.mean()
                        
                        # 分离的 advantage
                        advantage_init = mb_reward_init - baseline_init  # y1 优化自身准确率
                        advantage_corr = r2_tilde - baseline_corr        # y2 优化纠错+bonus
                    
                    # Policy Gradient: 分离优化
                    # y1: 最大化首轮准确率
                    loss_pg_init = -(advantage_init * sum_lp_init).mean()
                    # y2: 最大化纠错准确率 + 纠错增益奖励
                    loss_pg_corr = -(advantage_corr * sum_lp_corr).mean()
                    
                    # KL 正则化（长度归一化版本，与 Stage I 保持一致）
                    valid_mask_init = (~mb_mask_init).float()
                    valid_mask_corr = (~mb_mask_corr).float()
                    
                    kl_init_per_token = (logprob_init - mb_ref_init_logprob) * valid_mask_init
                    kl_corr_per_token = (logprob_corr - mb_ref_corr_logprob) * valid_mask_corr
                    
                    len_init = valid_mask_init.sum(dim=1).clamp(min=1.0)
                    len_corr = valid_mask_corr.sum(dim=1).clamp(min=1.0)
                    
                    kl_init_per_sample = kl_init_per_token.sum(dim=1) / len_init
                    kl_corr_per_sample = kl_corr_per_token.sum(dim=1) / len_corr
                    
                    kl_init_per_sample = torch.clamp(kl_init_per_sample, min=0.0)
                    kl_corr_per_sample = torch.clamp(kl_corr_per_sample, min=0.0)
                    
                    # KL 损失: β·(KL_1 + KL_2)
                    loss_kl_init = args.init_kl_coef * kl_init_per_sample.mean()
                    loss_kl_corr = args.corr_kl_coef * kl_corr_per_sample.mean()
                    loss_kl = loss_kl_init + loss_kl_corr
                    
                    # 总损失: 论文公式的负数形式
                    # -max E[r(y1) + r̃(y2)] + β·(KL_1 + KL_2)
                    # = min -E[r(y1)] - E[r̃(y2)] + β·(KL_1 + KL_2)
                    loss = loss_pg_init + loss_pg_corr + loss_kl
                    
                    # 记录用于日志
                    total_kl_init += kl_init_per_sample.detach().mean().item()
                    total_kl_corr += kl_corr_per_sample.detach().mean().item()
                loss = loss / self.algo_config['gradient_accumulation_steps']
                total_loss += loss.item()

                accelerator.backward(loss)
                micro_step += 1

                if micro_step == self.algo_config['gradient_accumulation_steps']:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    micro_step = 0

            # ------------------------
            # Logging / stats
            # ------------------------
            with torch.no_grad():
                mean_reward_corr = accelerator.gather_for_metrics(reward_corr).mean().item()
                mean_reward_init = accelerator.gather_for_metrics(reward_init).mean().item()
                mean_reward_delta = mean_reward_corr - mean_reward_init
                if 'reward_corr' not in ema_stats:
                    ema_stats['reward_corr'] = mean_reward_corr
                    ema_stats['reward_delta'] = mean_reward_delta
                    ema_stats['kl_corr'] = total_kl_corr / max(1, args.local_batch_size // args.per_device_train_batch_size)
                else:
                    # 只有当字典里已经有值了，才进行平滑更新
                    curr_kl = total_kl_corr / max(1, args.local_batch_size // args.per_device_train_batch_size)
                    ema_stats['reward_corr'] = (1 - ema_alpha) * ema_stats['reward_corr'] + ema_alpha * mean_reward_corr
                    ema_stats['reward_delta'] = (1 - ema_alpha) * ema_stats['reward_delta'] + ema_alpha * mean_reward_delta
                    ema_stats['kl_corr'] = (1 - ema_alpha) * ema_stats['kl_corr'] + ema_alpha * curr_kl

                metrics = {}
                metrics["score/kl_init"] = total_kl_init / max(1, args.local_batch_size // args.per_device_train_batch_size)
                metrics["score/kl_corr"] = total_kl_corr / max(1, args.local_batch_size // args.per_device_train_batch_size)
                metrics["score/reward_init"] = mean_reward_init
                metrics["score/reward_corr"] = mean_reward_corr
                metrics["score/reward_delta"] = mean_reward_corr - mean_reward_init  # 关键指标：纠错提升
                metrics["loss"] = total_loss
                metrics["episode"] = self.state.episode
                metrics["step"] = step_idx
                metrics["score_ema/reward_corr"] = ema_stats['reward_corr']
                metrics["score_ema/reward_delta"] = ema_stats['reward_delta']
                metrics["score_ema/kl_corr"] = ema_stats['kl_corr']
                self.log(metrics)

            del corr_outputs, corr_tokens, init_outputs, init_answers, queries
            del out_corr, out_init, logits_corr, logits_init, logprob_corr, logprob_init
            del ref_init_logprob, ref_corr_logprob, mask_init, mask_corr
            torch.cuda.empty_cache()
            gc.collect()

            self.state.global_step += 1
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(self.model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)

            # if (
            #     args.num_sample_generations > 0
            #     and (step_idx - 1) % max(1, args.num_total_batches // args.num_sample_generations) == 0
            # ):
            #     self.generate_completions(sampling=True)

            # Early termination if needed
            if self.control.should_training_stop:
                break

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(self.model, trial=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

        print("SCoRE training completed!")

    def generate_completions(self, sampling: bool = False):
        """
        Utility function to sample model completions on `eval_dataloader` and log them.
        Copied from RLOOTrainer's style, but simplified.
        """

        raise NotImplementedError('Not Yet')

        # args = self.args
        # if self.eval_dataloader is None:
        #     return

        # generation_config = GenerationConfig(
        #     max_new_tokens=args.response_length,
        #     temperature=(0.01 + 1e-7),
        #     top_k=0.0,
        #     top_p=1.0,
        #     do_sample=True,
        # )

        # table = defaultdict(list)
        # with unwrap_model_for_generation(
        #     self.model, self.accelerator, gather_deepspeed3_params=args.ds3_gather_for_generation
        # ) as unwrapped_model:
        #     for batch in self.eval_dataloader:
        #         query = batch["input_ids"]
        #         with torch.no_grad():
        #             context_length = query.shape[1]
        #             query_response, _ = batch_generation(
        #                 unwrapped_model,
        #                 query,
        #                 query.shape[0],
        #                 self.processing_class.pad_token_id,
        #                 generation_config,
        #             )
        #             response = query_response[:, context_length:]
        #             table["query"].extend(
        #                 gather_object(self.processing_class.batch_decode(query, skip_special_tokens=True))
        #             )
        #             table["model response"].extend(
        #                 gather_object(self.processing_class.batch_decode(response, skip_special_tokens=True))
        #             )

        #         if sampling:
        #             # Just do one batch if sampling
        #             break

        # df = pd.DataFrame(table)
        # if self.accelerator.is_main_process:
        #     print_rich_table(df.iloc[0 : 0 + 5])
        #     # If using W&B or Comet, you can log the table
        #     # ...
    
    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        try:
            import wandb
            from wandb import run as wandb_run
            wandb_url = wandb_run.get_url() if wandb_run is not None else None
        except ImportError:
            wandb_url = None

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb_url,
            comet_url=get_comet_experiment_url(),
            trainer_name="SCoRE",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))



def build_correction_inputs_for_batch(
    batch,
    init_answer_texts,
    tokenizer,
    prompt_builder,
    question_col: str = "question",
    initial_answer_col: str = "initial_answer",
):
    # We will store the final "correction input" for each row
    batch_correction_inputs = []

    for i, init_ans_text in enumerate(init_answer_texts):
        question_text = batch[question_col][i]

        # Build a 'sample' dict as your prompt_builder expects:
        sample_for_prompt = {
            question_col: question_text,
            initial_answer_col: [init_ans_text],  # your builder uses a list for initial answers
        }

        # Now get the final correction prompt(s). Typically there's 1 prompt
        # in this scenario, but build_correction_prompt returns a list.
        corr_inputs = prompt_builder.build_correction_prompt(
            sample=sample_for_prompt,
            tokenizer=tokenizer,
            question_col=question_col,
            initial_answer_col=initial_answer_col,
            tokenize=True
        )
        # Since we used only 1 initial answer, corr_prompts[0] is the final text
        corr_inputs = corr_inputs[0]

        batch_correction_inputs.append({'input_ids': corr_inputs})


    collated_corrections = DataCollatorWithPadding(tokenizer)(batch_correction_inputs)
    return collated_corrections['input_ids']
