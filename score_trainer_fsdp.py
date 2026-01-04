# score_trainer_fsdp_paper_stable.py
# Paper-aligned + stable SCoRETrainer (minimal changes from your original)
# Key changes:
# 1) Stage I: y1 is on-policy (sampled from current policy), and we apply strong KL(pi||ref) on y1 (paper Eq.3 spirit)
# 2) EMA baseline updates use GLOBAL mean across ranks
# 3) Advantage normalization uses GLOBAL std across ranks
# 4) Keep reward-delta clamp; do NOT hard-clip r2_tilde to tiny range

import gc
import math
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast
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
    get_cosine_schedule_with_warmup,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback

# trl utilities
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import (
    OnlineTrainerState,
    batch_generation,
    disable_dropout_in_model,
    forward,
    prepare_deepspeed,
    selective_log_softmax,
    generate_model_card,
    get_comet_experiment_url,
)
from trl.trainer.rloo_config import RLOOConfig

INVALID_LOGPROB = 1.0


@dataclass
class SCoREConfig(RLOOConfig):
    """
    Extended config for SCoRE (inherits RLOOConfig).
    - Stage I: strong KL on y1 uses beta2_kl
    - Stage II: weak KL on y1 and y2 uses beta1_kl
    - stage2_alpha: α for r2_tilde = r2 + α * clamp(r2-r1)
    - ema_alpha: smoothing factor for EMA baseline
    """
    stage: int = field(default=1, metadata={"help": "Stage of SCoRE Training (1 or 2)"})
    beta2_kl: float = field(default=0.2, metadata={"help": "β₂: strong KL(pi||ref) on y1 in Stage I"})
    beta1_kl: float = field(default=0.01, metadata={"help": "β₁: weak KL on y1 and y2 in Stage II"})
    stage2_alpha: float = field(default=0.5, metadata={"help": "α for r2_tilde = r2 + α (r2 - r1) (with clamp on delta)"})
    ema_alpha: float = field(default=0.03, metadata={"help": "EMA smoothing factor for baseline"})
    eval_reward_threshold: float = field(default=0.5, metadata={"help": "threshold for eval correctness if eval used"})
    max_eval_batches: int = field(default=64, metadata={"help": "max eval batches for light eval if called"})
    debug_metrics: bool = field(default=False, metadata={"help": "Enable debug metrics logging"})
    debug_metrics_interval: int = field(default=50, metadata={"help": "Interval for logging debug metrics"})

def compute_kl_robust(logprob: torch.Tensor, ref_logprob: torch.Tensor, mask: torch.Tensor):
    """
    Robust KL proxy estimate per-sample.
      - mask: boolean mask where True indicates padding token
      - token-wise diff -> clip -> per-sample mean -> diagnostics -> softplus for non-negativity

    NOTE:
      This is a KL *proxy* (sample-based estimate). It can be negative due to estimation noise.
      softplus makes it non-negative for optimization stability (as in your original).

    Returns:
      kl_final: tensor [B] (non-negative)
      percent_negative: float
      mean_raw_kl: float
    """
    CLIP = 20.0
    valid_mask = (~mask).float()  # 1 for valid tokens
    kl_per_token = torch.clamp(logprob - ref_logprob, min=-CLIP, max=CLIP) * valid_mask
    seq_len = valid_mask.sum(dim=1).clamp(min=1.0)
    mean_raw = kl_per_token.sum(dim=1) / seq_len
    percent_negative = (mean_raw < 0).float().mean().item()
    mean_raw_kl = mean_raw.mean().item()
    # kl_final = F.softplus(mean_raw)
    return mean_raw, percent_negative, mean_raw_kl

def compute_kl_sum_robust(logprob: torch.Tensor, ref_logprob: torch.Tensor, mask: torch.Tensor):
    """
    Same as compute_kl_robust but returns the SUM of KL over tokens for PG-style loss.
    """
    CLIP = 20.0
    valid_mask = (~mask).float()
    kl_per_token = torch.clamp(logprob - ref_logprob, min=-CLIP, max=CLIP) * valid_mask
    sum_kl = kl_per_token.sum(dim=1)
    
    # diagnostics
    seq_len = valid_mask.sum(dim=1).clamp(min=1.0)
    mean_raw = sum_kl / seq_len
    percent_negative = (mean_raw < 0).float().mean().item()
    mean_raw_kl = mean_raw.mean().item()
    
    return sum_kl, percent_negative, mean_raw_kl


class SCoRETrainer(Trainer):
    _tag_names = ["trl", "score"]

    def __init__(
        self,
        config: SCoREConfig,
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
        if ref_policy is policy:
            raise ValueError("`policy` and `ref_policy` cannot be the same object.")

        # freeze ref_policy
        for param in ref_policy.parameters():
            param.requires_grad = False
        ref_policy.eval()

        self.args = config
        args = config
        self.algo_config = algo_config
        self.processing_class = processing_class
        self.policy = policy
        self.ref_policy = ref_policy
        self.reward_model = reward_model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.optimizer, self.lr_scheduler = optimizers
        self.prompt_builder = prompt_builder

        if data_collator is None:
            data_collator = DataCollatorWithPadding(self.processing_class)
        self.data_collator = data_collator

        # disable dropout everywhere
        for module in [policy, ref_policy, reward_model]:
            if isinstance(module, nn.Module):
                disable_dropout_in_model(module)

        # token ids
        pad_id = getattr(self.processing_class, "pad_token_id", None)
        eos_id = getattr(self.processing_class, "eos_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.processing_class, "eos_token_id", None)
        if eos_id is None:
            eos_id = getattr(self.processing_class, "eos_token_id", None)

        self.init_generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            temperature=args.temperature,
            top_p=0.8,
            do_sample=True,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
        self.corr_generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            temperature=args.temperature,
            top_p=0.8,
            do_sample=True,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )

        # dataset length & episodes
        self.train_dataset_len = len(train_dataset)
        if args.total_episodes is None:
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)

        # Accelerator
        accelerator = Accelerator()
        self.accelerator = accelerator

        # ensure gradient_accumulation_steps present
        self.algo_config["gradient_accumulation_steps"] = self.algo_config.get("gradient_accumulation_steps", 1)
        try:
            accelerator.state.deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"] = self.algo_config[
                "gradient_accumulation_steps"
            ]
        except Exception:
            pass

        args.world_size = accelerator.num_processes

        # Batch size computations
        args.local_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.num_total_batches = math.ceil(args.total_episodes / args.batch_size)

        # run name + seed
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        _ = broadcast(time_tensor, 0).item()
        args.run_name = f"{args.exp_name}"

        self.local_seed = args.seed
        torch.manual_seed(args.seed)

        # DataLoader
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=args.local_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True,
        )

        # prepare model/optimizer/dataloader with Accelerator
        self.model = policy
        self.model, self.optimizer, self.dataloader = accelerator.prepare(self.model, self.optimizer, self.dataloader)

        # scheduler
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.algo_config.get("num_warmup_steps", 0),
            num_training_steps=args.num_total_batches,
        )

        # reset seed
        torch.manual_seed(self.local_seed)

        # eval dataloader optional
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

        # DeepSpeed / FSDP handling
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None

        if self.is_deepspeed_enabled:
            if isinstance(self.reward_model, nn.Module):
                self.reward_model = prepare_deepspeed(
                    self.reward_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )
            self.ref_policy = prepare_deepspeed(self.ref_policy, args.per_device_train_batch_size, args.fp16, args.bf16)
            self.deepspeed = self.model
        else:
            self.ref_policy = self.ref_policy.to(self.accelerator.device)
            if isinstance(self.reward_model, nn.Module):
                self.reward_model = self.reward_model.to(self.accelerator.device)

        # create optimizer if not provided
        if self.optimizer is None:
            self.create_optimizer_and_scheduler(num_training_steps=args.num_total_batches)

        # callbacks
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(self.callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler)
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)

        # trainer state
        self.control = TrainerControl()
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb
                for cb in self.callback_handler.callbacks + [self.control]
                if isinstance(cb, ExportableState)
            ],
            save_steps=args.save_steps,
        )
        self.current_flos = 0
        self.hp_search_backend = None

        # create output dir
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        # model tags
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        # EMA baseline initialization
        self.ema_baseline_init = None
        self.ema_baseline_corr = None
        self.ema_baseline_r2_tilde = None
        self.ema_alpha = args.ema_alpha

        # ✅ MA（长期平均）
        self.ma_reward_init_sum = 0
        self.ma_reward_corr_sum = 0
        self.ma_count = 0

        print(f"[INFO] Using EMA baseline with alpha={self.ema_alpha} (GLOBAL mean/std for multi-GPU stability)")

    def _batch_score_reward(self, outputs_texts, references):
        """
        Batch scoring wrapper for reward model.
        """
        device = self.accelerator.device
        refs = list(references)
        try:
            if hasattr(self.reward_model, "batch_score"):
                scores = self.reward_model.batch_score(outputs_texts, refs)
                return torch.tensor(scores, dtype=torch.float, device=device)
            if hasattr(self.reward_model, "score_batch"):
                scores = self.reward_model.score_batch(outputs_texts, refs)
                return torch.tensor(scores, dtype=torch.float, device=device)
            scores = self.reward_model(outputs_texts, refs)
            if isinstance(scores, torch.Tensor):
                return scores.to(device).float()
            return torch.tensor(list(scores), dtype=torch.float, device=device)
        except Exception:
            scores = []
            for out, ref in zip(outputs_texts, refs):
                scores.append(self.reward_model(model_answer=out, ground_truth=ref))
            return torch.tensor(scores, dtype=torch.float, device=device)

    def compute_ema_baseline(
        self,
        rewards: torch.Tensor,
        reward_type: str = "init",
        global_mean_val: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute EMA baseline for given rewards.
        IMPORTANT: when running multi-GPU, pass global_mean_val from gathered rewards.
        """
        mean_reward = float(global_mean_val) if global_mean_val is not None else rewards.mean().item()

        if reward_type == "init":
            if self.ema_baseline_init is None:
                self.ema_baseline_init = mean_reward
            else:
                self.ema_baseline_init = (1 - self.ema_alpha) * self.ema_baseline_init + self.ema_alpha * mean_reward
            return torch.full_like(rewards, self.ema_baseline_init)

        elif reward_type == "corr":
            if self.ema_baseline_corr is None:
                self.ema_baseline_corr = mean_reward
            else:
                self.ema_baseline_corr = (1 - self.ema_alpha) * self.ema_baseline_corr + self.ema_alpha * mean_reward
            return torch.full_like(rewards, self.ema_baseline_corr)

        elif reward_type == "r2_tilde":
            if self.ema_baseline_r2_tilde is None:
                self.ema_baseline_r2_tilde = mean_reward
            else:
                self.ema_baseline_r2_tilde = (1 - self.ema_alpha) * self.ema_baseline_r2_tilde + self.ema_alpha * mean_reward
            return torch.full_like(rewards, self.ema_baseline_r2_tilde)

        else:
            return torch.zeros_like(rewards)

    @torch.no_grad()
    def compute_eval_metrics(self, max_batches: int = 64, reward_threshold: float = 0.5):
        """
        Optional light-weight evaluation.
        """
        if self.eval_dataloader is None:
            return {}

        device = self.accelerator.device
        self.model.eval()

        n_seen = 0
        init_correct_list = []
        corr_correct_list = []
        for batch_idx, batch in enumerate(self.eval_dataloader):
            if batch_idx >= max_batches:
                break
            queries = batch["input_ids"].to(device).long()
            with unwrap_model_for_generation(
                self.model,
                self.accelerator,
                gather_deepspeed3_params=getattr(self.args, "ds3_gather_for_generation", False),
            ) as unwrapped_model:
                init_outputs, _ = batch_generation(
                    unwrapped_model,
                    queries,
                    self.args.local_rollout_forward_batch_size,
                    self.processing_class.pad_token_id,
                    self.init_generation_config,
                )
                init_context_len = queries.shape[1]
                init_answers = init_outputs[:, init_context_len:]
                init_texts = self.processing_class.batch_decode(init_answers, skip_special_tokens=True)

                corr_inputs = build_correction_inputs_for_batch(
                    batch,
                    init_texts,
                    self.processing_class,
                    self.prompt_builder,
                    question_col=self.algo_config["question_col"],
                ).to(device)
                
                corr_outputs, _ = batch_generation(
                    unwrapped_model,
                    corr_inputs,
                    self.args.local_rollout_forward_batch_size,
                    self.processing_class.pad_token_id,
                    self.corr_generation_config,
                )
                corr_context_len = corr_inputs.shape[1]
                corr_tokens = corr_outputs[:, corr_context_len:]
                corr_texts = self.processing_class.batch_decode(corr_tokens, skip_special_tokens=True)

            scores_init = self._batch_score_reward(init_texts, batch[self.algo_config["gold_col"]])
            scores_corr = self._batch_score_reward(corr_texts, batch[self.algo_config["gold_col"]])

            init_correct = (scores_init >= reward_threshold).float().cpu().numpy()
            corr_correct = (scores_corr >= reward_threshold).float().cpu().numpy()

            init_correct_list.append(init_correct)
            corr_correct_list.append(corr_correct)
            n_seen += init_correct.shape[0]

        if n_seen == 0:
            self.model.train()
            return {}

        init_all = np.concatenate(init_correct_list, axis=0)
        corr_all = np.concatenate(corr_correct_list, axis=0)
        acc_t1 = float(init_all.mean())
        acc_t2 = float(corr_all.mean())
        delta_t = acc_t2 - acc_t1
        delta_i_to_c = float(((init_all == 0) & (corr_all == 1)).sum() / len(init_all))
        delta_c_to_i = float(((init_all == 1) & (corr_all == 0)).sum() / len(init_all))

        metrics = {
            "eval/accuracy_t1": acc_t1,
            "eval/accuracy_t2": acc_t2,
            "eval/delta_t1_t2": delta_t,
            "eval/delta_i_to_c": delta_i_to_c,
            "eval/delta_c_to_i": delta_c_to_i,
            "eval/samples": int(len(init_all)),
        }
        self.model.train()
        return metrics

    def train(self):
        """
        Memory-friendly SCoRE training loop with EMA baseline:
         - Stage I (paper-aligned): on-policy y1 + strong KL(pi||ref) on y1 + PG on y2
         - Stage II: PG on y1 and y2 with weak KL constraints + progress reward shaping
        """
        args = self.args
        accelerator = self.accelerator
        device = accelerator.device
        dataloader = self.dataloader

        assert args.local_batch_size % args.per_device_train_batch_size == 0, (
            f"local_batch_size ({args.local_batch_size}) must be divisible by per_device_train_batch_size ({args.per_device_train_batch_size})."
        )
        num_micro_batches = args.local_batch_size // args.per_device_train_batch_size

        # trainer state
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches
        self.state.num_train_epochs = args.num_train_epochs

        # callbacks begin
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
        _start_time = time.time()

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())
        self.model.train()

        ema_stats = {}
        ema_alpha = 0.03

        # ensure tokenizer pad token set
        self.processing_class.padding_side = "left"
        if self.processing_class.pad_token_id is None:
            self.processing_class.pad_token_id = self.processing_class.eos_token_id
        pad_id = self.processing_class.pad_token_id

        # ====== helpers ======
        def global_mean(x: torch.Tensor) -> torch.Tensor:
            return accelerator.gather_for_metrics(x).mean()

        def global_std(x: torch.Tensor, min_std: float = 0.1) -> torch.Tensor:
            g = accelerator.gather_for_metrics(x)
            std = g.std()
            return std.clamp(min=min_std)  # ✅ 防止除零

        def logits_slice_for_generated(out_logits, outputs, context_len):
            logits_nxt = out_logits[:, :-1, :]
            gen_len = outputs.shape[1] - context_len
            start = context_len - 1
            end = start + gen_len
            assert end <= logits_nxt.shape[1], f"end ({end}) > logits_nxt_len ({logits_nxt.shape[1]})"
            return logits_nxt[:, start:end, :]

        def compute_ref_logprobs_chunked(ref_model, all_outputs, context_len, chunk_size):
            B, full_len = all_outputs.shape
            gen_len = full_len - context_len
            if gen_len <= 0:
                return torch.zeros((B, 0), device=device), torch.zeros((B, 0), dtype=torch.bool, device=device)

            ref_lp = torch.zeros((B, gen_len), dtype=torch.float, device=device)
            mask = torch.zeros((B, gen_len), dtype=torch.bool, device=device)

            for i in range(0, B, chunk_size):
                j = min(i + chunk_size, B)
                out_chunk = all_outputs[i:j]
                with torch.no_grad():
                    ref_out = forward(ref_model, out_chunk, pad_id)
                    logits_ref = logits_slice_for_generated(ref_out.logits, out_chunk, context_len)
                    y_tokens = out_chunk[:, context_len:]
                    lp_chunk = selective_log_softmax(logits_ref, y_tokens)
                    mask_chunk = (y_tokens == pad_id)
                    lp_chunk = lp_chunk.masked_fill_(mask_chunk, 0.0)
                    ref_lp[i:j] = lp_chunk
                    mask[i:j] = mask_chunk
                    del ref_out, logits_ref, lp_chunk, mask_chunk
                    torch.cuda.empty_cache()
            return ref_lp, mask

        # ====== main loop ======
        for step_idx in range(1, args.num_total_batches + 1):
            data = next(iter_dataloader)
            self.state.episode += args.batch_size

            # --------------------------
            # 1) Sampling y1 and y2 (no grad)
            # --------------------------
            with torch.no_grad():
                queries = data["input_ids"].to(device).long()

                with unwrap_model_for_generation(
                    self.model,
                    accelerator,
                    gather_deepspeed3_params=getattr(args, "ds3_gather_for_generation", False),
                ) as unwrapped_model:
                    # ✅ paper-aligned: y1 is on-policy in BOTH stages
                    init_outputs, _ = batch_generation(
                        unwrapped_model,
                        queries,
                        args.local_rollout_forward_batch_size,
                        pad_id,
                        self.init_generation_config,
                    )

                    init_context_len = queries.shape[1]
                    init_answers = init_outputs[:, init_context_len:]
                    init_answer_texts = self.processing_class.batch_decode(init_answers, skip_special_tokens=True)

                    # build correction inputs
                    corr_inputs = build_correction_inputs_for_batch(
                        data,
                        init_answer_texts,
                        self.processing_class,
                        self.prompt_builder,
                        question_col=self.algo_config["question_col"],
                    ).to(device)

                    # y2 always from current policy
                    corr_outputs, _ = batch_generation(
                        unwrapped_model,
                        corr_inputs,
                        args.local_rollout_forward_batch_size,
                        pad_id,
                        self.corr_generation_config,
                    )
                    corr_context_len = corr_inputs.shape[1]
                    corr_tokens = corr_outputs[:, corr_context_len:]

            # --------------------------
            # 2) ref logprobs (chunked)
            # --------------------------
            chunk_size = args.local_rollout_forward_batch_size if args.local_rollout_forward_batch_size > 0 else 1
            ref_init_logprob, mask_init = compute_ref_logprobs_chunked(self.ref_policy, init_outputs, init_context_len, chunk_size)
            ref_corr_logprob, mask_corr = compute_ref_logprobs_chunked(self.ref_policy, corr_outputs, corr_context_len, chunk_size)
            torch.cuda.empty_cache()

            # --------------------------
            # 3) rewards (batch)
            # --------------------------
            corr_output_text = self.processing_class.batch_decode(corr_tokens, skip_special_tokens=True)
            init_answer_texts_local = init_answer_texts
            with torch.no_grad():
                reward_corr = self._batch_score_reward(corr_output_text, data[self.algo_config["gold_col"]])
                reward_init = self._batch_score_reward(init_answer_texts_local, data[self.algo_config["gold_col"]])
            torch.cuda.empty_cache()
            if step_idx % 100 == 0:
                gc.collect()

            # --------------------------
            # 4) EMA Baselines (GLOBAL stats)
            # --------------------------
            with torch.no_grad():
                mean_init_global = global_mean(reward_init).item()
                mean_corr_global = global_mean(reward_corr).item()

                baseline_init = self.compute_ema_baseline(reward_init, "init", global_mean_val=mean_init_global)
                baseline_corr = self.compute_ema_baseline(reward_corr, "corr", global_mean_val=mean_corr_global)

                if args.stage == 2:
                    # progress shaping (paper): r2_tilde = r2 + α * (r2 - r1)
                    # keep your clamp on delta for stability
                    reward_delta = torch.clamp(reward_corr - reward_init, min=-0.5, max=0.5)
                    r2_tilde = reward_corr + args.stage2_alpha * reward_delta

                    mean_r2_tilde_global = global_mean(r2_tilde).item()
                    baseline_r2_tilde = self.compute_ema_baseline(r2_tilde, "r2_tilde", global_mean_val=mean_r2_tilde_global)

            # MA stats
            self.ma_reward_init_sum += reward_init.sum().item()
            self.ma_reward_corr_sum += reward_corr.sum().item()
            self.ma_count += reward_init.shape[0]

            # --------------------------
            # 5) Loss & gradient (micro-batches)
            # --------------------------
            total_loss = 0.0
            total_kl_init = 0.0
            total_kl_corr = 0.0
            total_logprob_init = 0.0

            percent_neg_init = 0.0
            percent_neg_corr = 0.0
            raw_kl_init = 0.0
            raw_kl_corr = 0.0

            # precompute GLOBAL std for advantage normalization (stable on multi-GPU)
            with torch.no_grad():
                if args.stage == 1:
                    adv_corr_full = reward_corr - baseline_corr
                    adv_corr_std = global_std(adv_corr_full, min_std=0.1)
                else:
                    adv_init_full = reward_init - baseline_init
                    adv_init_std = global_std(adv_init_full, min_std=0.1)

                    adv_corr_full = r2_tilde - baseline_r2_tilde
                    adv_corr_std = global_std(adv_corr_full, min_std=0.1)

            for micro_idx, micro_start in enumerate(range(0, args.local_batch_size, args.per_device_train_batch_size)):
                micro_end = micro_start + args.per_device_train_batch_size
                mb_idx = slice(micro_start, micro_end)

                mb_corr_outputs = corr_outputs[mb_idx]
                mb_corr_tokens = corr_tokens[mb_idx]
                mb_init_outputs = init_outputs[mb_idx]
                mb_init_tokens = init_answers[mb_idx]
                mb_ref_init_logprob = ref_init_logprob[mb_idx]
                mb_ref_corr_logprob = ref_corr_logprob[mb_idx]
                mb_mask_init = mask_init[mb_idx]
                mb_mask_corr = mask_corr[mb_idx]
                mb_queries = queries[mb_idx]
                mb_init_context_len = mb_queries.shape[1]

                # forward model for initial/correction outputs (with gradients)
                out_init = forward(self.model, mb_init_outputs, pad_id)
                logits_init_for_gen = logits_slice_for_generated(out_init.logits, mb_init_outputs, mb_init_context_len)
                logprob_init = selective_log_softmax(logits_init_for_gen, mb_init_tokens)
                logprob_init = logprob_init.masked_fill_(mb_mask_init, 0.0)
                del out_init, logits_init_for_gen
                torch.cuda.empty_cache()

                out_corr = forward(self.model, mb_corr_outputs, pad_id)
                logits_corr_for_gen = logits_slice_for_generated(out_corr.logits, mb_corr_outputs, corr_context_len)
                logprob_corr = selective_log_softmax(logits_corr_for_gen, mb_corr_tokens)
                logprob_corr = logprob_corr.masked_fill_(mb_mask_corr, 0.0)
                del out_corr, logits_corr_for_gen
                torch.cuda.empty_cache()

                sum_lp_corr = logprob_corr.sum(dim=1)
                sum_lp_init = logprob_init.sum(dim=1)

                if args.stage == 1:
                    # ✅ Stage I (paper-aligned):
                    # - PG on y2 using reward_corr (with KL penalty)
                    # - Strong KL(pi||ref) on y1 using beta2_kl (as PG-style centering force)
                    
                    kl_init_sum, pct_neg_init, raw_init = compute_kl_sum_robust(
                        logprob_init, mb_ref_init_logprob, mb_mask_init
                    )
                    kl_corr_sum, pct_neg_corr, raw_corr = compute_kl_sum_robust(
                        logprob_corr, mb_ref_corr_logprob, mb_mask_corr
                    )

                    adv_corr_mb = (reward_corr - baseline_corr)[mb_idx]
                    adv_corr_mb = adv_corr_mb / adv_corr_std
                    # Add weak KL penalty to y2 advantage
                    adv_corr_mb = adv_corr_mb - args.beta1_kl * kl_corr_sum
                    adv_corr_mb = torch.clamp(adv_corr_mb, min=-2.0, max=2.0)
                    
                    # PG on y2
                    loss_pg_corr = -(adv_corr_mb.detach() * sum_lp_corr).mean()

                    # Strong KL on y1: use PG-style centering force
                    # This is more stable than a linear penalty as it pushes back harder when far
                    adv_init_mb = - args.beta2_kl * kl_init_sum
                    adv_init_mb = torch.clamp(adv_init_mb, min=-2.0, max=2.0)
                    loss_kl_init_strong = -(adv_init_mb.detach() * sum_lp_init).mean()

                    loss = loss_pg_corr + loss_kl_init_strong

                    # ---- logging accumulators ----
                    total_kl_init += raw_init
                    total_kl_corr += raw_corr

                    percent_neg_init = pct_neg_init
                    raw_kl_init = raw_init
                    percent_neg_corr = pct_neg_corr
                    raw_kl_corr = raw_corr

                    # optional diag: average logprob init
                    total_logprob_init += sum_lp_init.mean().item()

                else:
                    # Stage II:
                    # - PG on y1 with reward_init
                    # - PG on y2 with r2_tilde
                    # - weak KL on both y1 and y2 with beta1_kl
                    
                    kl_init_sum, pct_neg_init, raw_init = compute_kl_sum_robust(
                        logprob_init, mb_ref_init_logprob, mb_mask_init
                    )
                    kl_corr_sum, pct_neg_corr, raw_corr = compute_kl_sum_robust(
                        logprob_corr, mb_ref_corr_logprob, mb_mask_corr
                    )

                    adv_init_mb = (reward_init - baseline_init)[mb_idx]
                    adv_init_mb = adv_init_mb / adv_init_std
                    adv_init_mb = adv_init_mb - args.beta1_kl * kl_init_sum
                    adv_init_mb = torch.clamp(adv_init_mb, min=-2.0, max=2.0)

                    adv_corr_mb = (r2_tilde - baseline_r2_tilde)[mb_idx]
                    adv_corr_mb = adv_corr_mb / adv_corr_std
                    adv_corr_mb = adv_corr_mb - args.beta1_kl * kl_corr_sum
                    adv_corr_mb = torch.clamp(adv_corr_mb, min=-2.0, max=2.0)

                    loss_pg_init = -(adv_init_mb.detach() * sum_lp_init).mean()
                    loss_pg_corr = -(adv_corr_mb.detach() * sum_lp_corr).mean()

                    loss = loss_pg_init + loss_pg_corr

                    total_kl_init += raw_init
                    total_kl_corr += raw_corr

                    percent_neg_init = pct_neg_init
                    percent_neg_corr = pct_neg_corr
                    raw_kl_init = raw_init
                    raw_kl_corr = raw_corr

                # gradient accumulation
                loss = loss / float(num_micro_batches)
                total_loss += float(loss.item())

                accelerator.backward(loss)

                # update on last micro-batch
                if micro_idx == num_micro_batches - 1:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                del logprob_init, logprob_corr
                torch.cuda.empty_cache()

            # --------------------------
            # 6) Logging
            # --------------------------
            # --------------------------
            # 6) Logging (core by default, debug optional)
            # --------------------------
            with torch.no_grad():
                mean_reward_corr = global_mean(reward_corr).item()
                mean_reward_init = global_mean(reward_init).item()
                mean_reward_delta = mean_reward_corr - mean_reward_init

                metrics = {
                    # axis
                    "step": step_idx,
                    "episode": self.state.episode,

                    # core training signals
                    "loss": total_loss,
                    "score/reward_init": mean_reward_init,
                    "score/reward_corr": mean_reward_corr,
                    "score/reward_delta": mean_reward_delta,
                    "score/kl_init": total_kl_init / max(1, num_micro_batches),
                    "score/kl_corr": total_kl_corr / max(1, num_micro_batches),

                    # EMA baselines (core, low-noise)
                    "baseline/ema_init": float(self.ema_baseline_init or 0.0),
                    "baseline/ema_corr": float(self.ema_baseline_corr or 0.0),
                }

                # stage-specific core hyperparams (small but useful)
                if args.stage == 1:
                    metrics["stage1/beta2_kl"] = args.beta2_kl
                    metrics["stage1/beta1_kl"] = args.beta1_kl
                else:
                    metrics["stage2/alpha"] = args.stage2_alpha
                    metrics["stage2/beta1_kl"] = args.beta1_kl
                    metrics["baseline/ema_r2_tilde"] = float(self.ema_baseline_r2_tilde or 0.0)

                # ---- optional debug metrics ----
                debug_on = bool(getattr(args, "debug_metrics", False))
                debug_every = int(getattr(args, "debug_metrics_interval", 50))
                if debug_on and (step_idx % max(1, debug_every) == 0):
                    # These are diagnostics only; do not affect training
                    metrics.update({
                        "debug/raw_kl_init": raw_kl_init,
                        "debug/raw_kl_corr": raw_kl_corr,
                        "debug/percent_negative_kl_init": percent_neg_init,
                        "debug/percent_negative_kl_corr": percent_neg_corr,
                    })

                    # Advantage std diagnostics: compute the right ones per stage
                    if args.stage == 1:
                        metrics["debug/std_adv_corr_local"] = (reward_corr - baseline_corr).std().item()
                    else:
                        metrics["debug/std_adv_init_local"] = (reward_init - baseline_init).std().item()
                        metrics["debug/std_adv_corr_local"] = (r2_tilde - baseline_r2_tilde).std().item()
                        metrics["debug/r2_tilde_mean"] = global_mean(r2_tilde).item()

                self.log(metrics)


            # cleanup per-batch big tensors
            try:
                del corr_outputs, corr_tokens, init_outputs, init_answers, queries
            except Exception:
                pass
            try:
                del ref_init_logprob, ref_corr_logprob, mask_init, mask_corr
            except Exception:
                pass
            torch.cuda.empty_cache()
            if step_idx % 100 == 0:
                gc.collect()

            # step end housekeeping
            self.state.global_step += 1
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(self.model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)

            if self.control.should_training_stop:
                break

        # final
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(self.model, trial=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

        # optional final light eval
        if self.eval_dataloader is not None:
            eval_metrics = self.compute_eval_metrics(
                max_batches=getattr(self.args, "max_eval_batches", 64),
                reward_threshold=getattr(self.args, "eval_reward_threshold", 0.5),
            )
            if eval_metrics:
                self.log(eval_metrics)

        print("SCoRE training completed!")

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        try:
            import wandb
            from wandb import run as wandb_run
            wandb_url = wandb_run.get_url() if wandb_run is not None else None
        except Exception:
            wandb_url = None

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=getattr(self, "hub_model_id", None),
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
    """
    Build correction inputs for a batch using prompt_builder.
    Returns padded input_ids tensor for correction prompts.
    """
    batch_correction_inputs = []
    for i, init_ans_text in enumerate(init_answer_texts):
        question_text = batch[question_col][i]
        sample_for_prompt = {
            question_col: question_text,
            initial_answer_col: init_ans_text,
        }
        corr_inputs = prompt_builder.build_correction_prompt(
            sample=sample_for_prompt,
            tokenizer=tokenizer,
            question_col=question_col,
            initial_answer_col=initial_answer_col,
            tokenize=True,
        )
        corr_inputs = corr_inputs[0]
        batch_correction_inputs.append({"input_ids": corr_inputs})

    collated_corrections = DataCollatorWithPadding(tokenizer)(batch_correction_inputs)
    return collated_corrections["input_ids"]
