import argparse
import csv
import json
import math
import os
import time
from typing import Optional, Union, List
from datetime import datetime

import numpy as np
import torch
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from alphagen.data.calculator import AlphaCalculator

from alphagen.data.expression import *
from alphagen.models.alpha_pool import AlphaPool, AlphaPoolBase
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet, TransformerSharedNet
from alphagen.utils.random import reseed_everything
from alphagen.rl.env.core import AlphaEnvCore
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.compat import patch_all
from alphagen.config import REWARD_PER_STEP


def log_metrics_csv(run_dir: Optional[str], step: int, metrics: dict) -> None:
    if not run_dir:
        return
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "metrics_step.csv")
    fieldnames = [
        "step",
        "pool_size",
        "best_ic",
        "best_rankic",
        "mean_ic",
        "mean_abs_ic",
        "weighted_mean_ic",
        "mean_rankic",
        "test_rankic",
        "reward_total",
        "RE",
        "RI_func",
        "RI_struct",
        "RI_reg",
        "turnover_mean",
        "turnover_penalty",
        "turnover_weight",
        "turnover_topk",
        "turnover_baseline",
        "turnover_step_stride",
        "reward_lambda_t",
        "ri_func_backend",
        "ri_struct_backend",
        "optimize_executed",
        "optimize_every",
        "optimize_n_iter",
        "cluster_id",
        "cluster_count",
        "cluster_mean_re",
        "cluster_positive_re_rate",
        "token_cluster_id",
        "token_cluster_count",
        "token_cluster_mean_re",
        "token_cluster_positive_re_rate",
        "ast_cluster_id",
        "ast_cluster_count",
        "ast_cluster_mean_re",
        "ast_cluster_positive_re_rate",
        "ast_cluster_similarity",
        "ast_cluster_is_new",
        "corr_cluster_id",
        "corr_cluster_count",
        "corr_cluster_mean_re",
        "corr_cluster_positive_re_rate",
        "corr_cluster_similarity",
        "corr_cluster_is_new",
        "ri_func_eval_ms",
        "ri_func_stack_ms",
        "ri_func_lstsq_ms",
        "ri_func_metric_ms",
        "ri_func_total_ms",
        "ri_func_used_k",
        "ri_func_used_sample",
        "alpha_cache_size",
        "alpha_cache_max_size",
        "alpha_cache_hits",
        "alpha_cache_misses",
        "alpha_cache_hit_rate",
        "single_ic_cache_size",
        "single_ic_cache_max_size",
        "single_ic_cache_hits",
        "single_ic_cache_misses",
        "single_ic_cache_hit_rate",
        "mutual_ic_cache_size",
        "mutual_ic_cache_max_size",
        "mutual_ic_cache_hits",
        "mutual_ic_cache_misses",
        "mutual_ic_cache_hit_rate",
        "profile_try_total_sec",
        "profile_calc_ics_sec",
        "profile_ri_func_sec",
        "profile_ri_struct_sec",
        "profile_ri_reg_sec",
        "profile_add_factor_sec",
        "profile_optimize_sec",
        "profile_pop_sec",
        "profile_eval_ensemble_sec",
        "profile_compose_reward_sec",
        "profile_invalid_expr_sec",
        "profile_tracked_total_sec",
        "profile_rollout_wall_sec",
        "profile_test_ensemble_sec",
        "profile_save_ckpt_sec",
        "profile_tracked_ratio",
    ]
    write_header = not os.path.isfile(path)
    row = {name: metrics.get(name, math.nan) for name in fieldnames}
    row["step"] = step
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


class CustomCallback(BaseCallback):
    def __init__(self,
                 save_freq: int,
                 show_freq: int,
                 save_path: str,
                 run_dir: Optional[str],
                 valid_calculator: AlphaCalculator,
                 test_calculator: AlphaCalculator,
                 save_model_ckpt: bool = False,
                 name_prefix: str = 'rl_model',
                 timestamp: Optional[str] = None,
                 verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.show_freq = show_freq
        self.save_path = save_path
        self.run_dir = run_dir
        self.name_prefix = name_prefix

        self.valid_calculator = valid_calculator
        self.test_calculator = test_calculator
        self.save_model_ckpt = bool(save_model_ckpt)
        self._last_rollout_ts: Optional[float] = None
        self._rollout_wall_total_sec: float = 0.0
        self._test_ensemble_total_sec: float = 0.0
        self._save_ckpt_total_sec: float = 0.0

        if timestamp is None:
            self.timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        else:
            self.timestamp = timestamp

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        if self.run_dir is not None:
            os.makedirs(self.run_dir, exist_ok=True)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        assert self.logger is not None
        now_ts = time.perf_counter()
        if self._last_rollout_ts is not None:
            self._rollout_wall_total_sec += max(0.0, now_ts - self._last_rollout_ts)
        self._last_rollout_ts = now_ts
        self.logger.record('pool/size', self.pool.size)
        self.logger.record('pool/significant', (np.abs(self.pool.weights[:self.pool.size]) > 1e-4).sum())
        self.logger.record('pool/best_ic_ret', self.pool.best_ic_ret)
        self.logger.record('pool/eval_cnt', self.pool.eval_cnt)
        t_test0 = time.perf_counter()
        ic_test, rank_ic_test = self.pool.test_ensemble(self.test_calculator)
        self._test_ensemble_total_sec += max(0.0, time.perf_counter() - t_test0)
        self.logger.record('test/ic', ic_test)
        self.logger.record('test/rank_ic', rank_ic_test)
        rank_ic_test_value = float(rank_ic_test) if rank_ic_test is not None else math.nan
        pool_size = getattr(self.pool, "size", math.nan)
        mean_abs_ic = math.nan
        weighted_mean_ic = math.nan
        if isinstance(pool_size, (int, np.integer)) and pool_size > 0 and hasattr(self.pool, "single_ics"):
            single_ics = np.asarray(self.pool.single_ics[:pool_size], dtype=float)
            mean_ic = float(np.nanmean(single_ics))
            mean_abs_ic = float(np.nanmean(np.abs(single_ics)))
            if hasattr(self.pool, "weights"):
                weights = np.asarray(self.pool.weights[:pool_size], dtype=float)
                valid_mask = ~np.isnan(single_ics) & ~np.isnan(weights)
                if valid_mask.any():
                    weights_valid = weights[valid_mask]
                    single_ics_valid = single_ics[valid_mask]
                    weight_l1 = float(np.sum(np.abs(weights_valid)))
                    if weight_l1 > 0.0:
                        weighted_mean_ic = float(np.dot(weights_valid, single_ics_valid) / weight_l1)
        else:
            mean_ic = math.nan
        profile = {}
        if hasattr(self.pool, "profile_snapshot"):
            try:
                profile = self.pool.profile_snapshot()  # type: ignore[attr-defined]
            except Exception:
                profile = {}
        totals = profile.get("timing_totals_sec", {}) if isinstance(profile, dict) else {}
        tracked_total_sec = float(profile.get("timing_tracked_total_sec", 0.0)) if isinstance(profile, dict) else 0.0
        t_save0 = time.perf_counter()
        self.save_checkpoint()
        self._save_ckpt_total_sec += max(0.0, time.perf_counter() - t_save0)
        tracked_with_cb = tracked_total_sec + self._test_ensemble_total_sec + self._save_ckpt_total_sec
        tracked_ratio = (
            tracked_with_cb / max(self._rollout_wall_total_sec, 1e-12)
            if self._rollout_wall_total_sec > 0 else math.nan
        )
        log_metrics_csv(
            self.run_dir or self.save_path,
            self.num_timesteps,
            {
                "pool_size": pool_size,
                "best_ic": getattr(self.pool, "best_ic_ret", math.nan),
                "best_rankic": rank_ic_test_value,
                "mean_ic": mean_ic,
                "mean_abs_ic": mean_abs_ic,
                "weighted_mean_ic": weighted_mean_ic,
                "mean_rankic": math.nan,
                "test_rankic": rank_ic_test_value,
                "reward_total": getattr(self.pool, "last_reward_info", {}).get("reward_total", math.nan),
                "RE": getattr(self.pool, "last_reward_info", {}).get("re", math.nan),
                "RI_func": getattr(self.pool, "last_reward_info", {}).get("ri_func", math.nan),
                "RI_struct": getattr(self.pool, "last_reward_info", {}).get("ri_struct", math.nan),
                "RI_reg": getattr(self.pool, "last_reward_info", {}).get("ri_reg", math.nan),
                "turnover_mean": getattr(self.pool, "last_reward_info", {}).get("turnover_mean", math.nan),
                "turnover_penalty": getattr(self.pool, "last_reward_info", {}).get("turnover_penalty", math.nan),
                "turnover_weight": getattr(self.pool, "last_reward_info", {}).get("turnover_weight", math.nan),
                "turnover_topk": getattr(self.pool, "last_reward_info", {}).get("turnover_topk", math.nan),
                "turnover_baseline": getattr(self.pool, "last_reward_info", {}).get("turnover_baseline", math.nan),
                "turnover_step_stride": getattr(self.pool, "last_reward_info", {}).get("turnover_step_stride", math.nan),
                "reward_lambda_t": getattr(self.pool, "last_reward_info", {}).get("reward_lambda_t", math.nan),
                "ri_func_backend": getattr(self.pool, "last_reward_info", {}).get("ri_func_backend", math.nan),
                "ri_struct_backend": getattr(self.pool, "last_reward_info", {}).get("ri_struct_backend", math.nan),
                "optimize_executed": getattr(self.pool, "last_reward_info", {}).get("optimize_executed", math.nan),
                "optimize_every": getattr(self.pool, "last_reward_info", {}).get("optimize_every", math.nan),
                "optimize_n_iter": getattr(self.pool, "last_reward_info", {}).get("optimize_n_iter", math.nan),
                "cluster_id": getattr(self.pool, "last_reward_info", {}).get("cluster_id", math.nan),
                "cluster_count": getattr(self.pool, "last_reward_info", {}).get("cluster_count", math.nan),
                "cluster_mean_re": getattr(self.pool, "last_reward_info", {}).get("cluster_mean_re", math.nan),
                "cluster_positive_re_rate": getattr(self.pool, "last_reward_info", {}).get("cluster_positive_re_rate", math.nan),
                "token_cluster_id": getattr(self.pool, "last_reward_info", {}).get("token_cluster_id", math.nan),
                "token_cluster_count": getattr(self.pool, "last_reward_info", {}).get("token_cluster_count", math.nan),
                "token_cluster_mean_re": getattr(self.pool, "last_reward_info", {}).get("token_cluster_mean_re", math.nan),
                "token_cluster_positive_re_rate": getattr(self.pool, "last_reward_info", {}).get("token_cluster_positive_re_rate", math.nan),
                "ast_cluster_id": getattr(self.pool, "last_reward_info", {}).get("ast_cluster_id", math.nan),
                "ast_cluster_count": getattr(self.pool, "last_reward_info", {}).get("ast_cluster_count", math.nan),
                "ast_cluster_mean_re": getattr(self.pool, "last_reward_info", {}).get("ast_cluster_mean_re", math.nan),
                "ast_cluster_positive_re_rate": getattr(self.pool, "last_reward_info", {}).get("ast_cluster_positive_re_rate", math.nan),
                "ast_cluster_similarity": getattr(self.pool, "last_reward_info", {}).get("ast_cluster_similarity", math.nan),
                "ast_cluster_is_new": getattr(self.pool, "last_reward_info", {}).get("ast_cluster_is_new", math.nan),
                "corr_cluster_id": getattr(self.pool, "last_reward_info", {}).get("corr_cluster_id", math.nan),
                "corr_cluster_count": getattr(self.pool, "last_reward_info", {}).get("corr_cluster_count", math.nan),
                "corr_cluster_mean_re": getattr(self.pool, "last_reward_info", {}).get("corr_cluster_mean_re", math.nan),
                "corr_cluster_positive_re_rate": getattr(self.pool, "last_reward_info", {}).get("corr_cluster_positive_re_rate", math.nan),
                "corr_cluster_similarity": getattr(self.pool, "last_reward_info", {}).get("corr_cluster_similarity", math.nan),
                "corr_cluster_is_new": getattr(self.pool, "last_reward_info", {}).get("corr_cluster_is_new", math.nan),
                "ri_func_eval_ms": getattr(self.pool, "last_reward_info", {}).get("ri_func_eval_ms", math.nan),
                "ri_func_stack_ms": getattr(self.pool, "last_reward_info", {}).get("ri_func_stack_ms", math.nan),
                "ri_func_lstsq_ms": getattr(self.pool, "last_reward_info", {}).get("ri_func_lstsq_ms", math.nan),
                "ri_func_metric_ms": getattr(self.pool, "last_reward_info", {}).get("ri_func_metric_ms", math.nan),
                "ri_func_total_ms": getattr(self.pool, "last_reward_info", {}).get("ri_func_total_ms", math.nan),
                "ri_func_used_k": getattr(self.pool, "last_reward_info", {}).get("ri_func_used_k", math.nan),
                "ri_func_used_sample": getattr(self.pool, "last_reward_info", {}).get("ri_func_used_sample", math.nan),
                "alpha_cache_size": getattr(self.pool, "last_reward_info", {}).get("alpha_cache_size", math.nan),
                "alpha_cache_max_size": getattr(self.pool, "last_reward_info", {}).get("alpha_cache_max_size", math.nan),
                "alpha_cache_hits": getattr(self.pool, "last_reward_info", {}).get("alpha_cache_hits", math.nan),
                "alpha_cache_misses": getattr(self.pool, "last_reward_info", {}).get("alpha_cache_misses", math.nan),
                "alpha_cache_hit_rate": getattr(self.pool, "last_reward_info", {}).get("alpha_cache_hit_rate", math.nan),
                "single_ic_cache_size": getattr(self.pool, "last_reward_info", {}).get("single_ic_cache_size", math.nan),
                "single_ic_cache_max_size": getattr(self.pool, "last_reward_info", {}).get("single_ic_cache_max_size", math.nan),
                "single_ic_cache_hits": getattr(self.pool, "last_reward_info", {}).get("single_ic_cache_hits", math.nan),
                "single_ic_cache_misses": getattr(self.pool, "last_reward_info", {}).get("single_ic_cache_misses", math.nan),
                "single_ic_cache_hit_rate": getattr(self.pool, "last_reward_info", {}).get("single_ic_cache_hit_rate", math.nan),
                "mutual_ic_cache_size": getattr(self.pool, "last_reward_info", {}).get("mutual_ic_cache_size", math.nan),
                "mutual_ic_cache_max_size": getattr(self.pool, "last_reward_info", {}).get("mutual_ic_cache_max_size", math.nan),
                "mutual_ic_cache_hits": getattr(self.pool, "last_reward_info", {}).get("mutual_ic_cache_hits", math.nan),
                "mutual_ic_cache_misses": getattr(self.pool, "last_reward_info", {}).get("mutual_ic_cache_misses", math.nan),
                "mutual_ic_cache_hit_rate": getattr(self.pool, "last_reward_info", {}).get("mutual_ic_cache_hit_rate", math.nan),
                "profile_try_total_sec": totals.get("try_new_expr_total_sec", math.nan),
                "profile_calc_ics_sec": totals.get("calc_ics_sec", math.nan),
                "profile_ri_func_sec": totals.get("ri_func_sec", math.nan),
                "profile_ri_struct_sec": totals.get("ri_struct_sec", math.nan),
                "profile_ri_reg_sec": totals.get("ri_reg_sec", math.nan),
                "profile_add_factor_sec": totals.get("add_factor_sec", math.nan),
                "profile_optimize_sec": totals.get("optimize_sec", math.nan),
                "profile_pop_sec": totals.get("pop_sec", math.nan),
                "profile_eval_ensemble_sec": totals.get("evaluate_ensemble_sec", math.nan),
                "profile_compose_reward_sec": totals.get("compose_reward_sec", math.nan),
                "profile_invalid_expr_sec": totals.get("invalid_expr_sec", math.nan),
                "profile_tracked_total_sec": tracked_with_cb,
                "profile_rollout_wall_sec": self._rollout_wall_total_sec,
                "profile_test_ensemble_sec": self._test_ensemble_total_sec,
                "profile_save_ckpt_sec": self._save_ckpt_total_sec,
                "profile_tracked_ratio": tracked_ratio,
            },
        )
        self._write_profile_summary()

    def save_checkpoint(self):
        path = os.path.join(self.save_path, f'{self.num_timesteps}_steps')
        if self.save_model_ckpt:
            self.model.save(path)   # type: ignore
            if self.verbose > 1:
                print(f'Saving model checkpoint to {path}')
        with open(f'{path}_pool.json', 'w') as f:
            json.dump(self.pool.to_dict(), f)

    def _write_profile_summary(self) -> None:
        out_dir = self.run_dir or self.save_path
        if not out_dir:
            return
        profile = {}
        if hasattr(self.pool, "profile_snapshot"):
            try:
                profile = self.pool.profile_snapshot()  # type: ignore[attr-defined]
            except Exception:
                profile = {}
        totals = profile.get("timing_totals_sec", {}) if isinstance(profile, dict) else {}
        tracked_pool_sec = float(profile.get("timing_tracked_total_sec", 0.0)) if isinstance(profile, dict) else 0.0
        tracked_all_sec = tracked_pool_sec + self._test_ensemble_total_sec + self._save_ckpt_total_sec
        rollout_wall = self._rollout_wall_total_sec
        ratio = tracked_all_sec / rollout_wall if rollout_wall > 0 else None
        components = dict(totals)
        components["test_ensemble_sec"] = self._test_ensemble_total_sec
        components["save_checkpoint_sec"] = self._save_ckpt_total_sec
        sorted_components = sorted(
            [{"name": k, "seconds": float(v)} for k, v in components.items()],
            key=lambda x: x["seconds"],
            reverse=True,
        )
        payload = {
            "run_id": os.path.basename(self.run_dir or ""),
            "timesteps": int(self.num_timesteps),
            "eval_cnt": int(getattr(self.pool, "eval_cnt", 0)),
            "tracked_pool_sec": tracked_pool_sec,
            "tracked_all_sec": tracked_all_sec,
            "rollout_wall_sec": rollout_wall,
            "tracked_ratio": ratio,
            "components": sorted_components,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "profile_summary.json"), "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def show_pool_state(self):
        state = self.pool.state
        n = len(state['exprs'])
        print('---------------------------------------------')
        for i in range(n):
            weight = state['weights'][i]
            expr_str = str(state['exprs'][i])
            ic_ret = state['ics_ret'][i]
            print(f'> Alpha #{i}: {weight}, {expr_str}, {ic_ret}')
        print(f'>> Ensemble ic_ret: {state["best_ic_ret"]}')
        print('---------------------------------------------')

    @property
    def pool(self) -> AlphaPoolBase:
        return self.env_core.pool

    @property
    def env_core(self) -> AlphaEnvCore:
        return self.training_env.envs[0].unwrapped  # type: ignore


def main(
    seed: int = 0,
    market: str = "csi300",
    pool_capacity: int = 50,
    steps: int = 200_000,
    backbone: str = "lstm",
    re_mode: str = "ensemble",
    reward_mode: str = "re",
    lambda_ri: float = 0.0,
    ri_func_weight: float = 1.0,
    ri_struct_weight: float = 1.0,
    ri_reg_weight: float = 1.0,
    ri_schedule_decay: float = 0.0,
    ri_struct_value_bonus: float = 0.1,
    ri_struct_underexplore_power: float = 1.0,
    ri_func_metric: str = "rankic",
    ri_func_topk: int = 8,
    ri_func_sample_size: int = 128,
    ri_func_rankic_on_cpu: bool = True,
    ri_admission_gate: bool = False,
    ri_func_backend: str = "auto",
    ri_struct_backend: str = "token_bigram",
    ri_corr_threshold: float = 0.8,
    ri_ast_similarity_threshold: float = 0.9,
    ri_corr_value_bonus: float = 0.1,
    ri_corr_underexplore_power: float = 1.0,
    ri_turnover_weight: float = 0.0,
    ri_turnover_topk: int = 30,
    ri_turnover_baseline: float = 0.5,
    ri_turnover_step_stride: int = 5,
    profile_timing: bool = True,
    optimize_every: int = 2,
    optimize_n_iter: int = 256,
    save_model_ckpt: bool = False,
    reward_per_step: float = REWARD_PER_STEP,
    ri_reg_l0: Optional[float] = None,
    ri_struct_topk: int = 5,
    run_name: str = "",
    logdir: str = "",
    provider_uri: str = "",
    ckpt_dir: str = "",
    tb_dir: str = "",
    device: str = "auto",
    verbose: int = 0,
):
    reseed_everything(seed)

    resolved_provider_uri = (
        provider_uri
        or os.environ.get("QLIB_PROVIDER_URI", "")
        or "/kaggle/input/baostock/cn_data_baostock_fwdadj"
    )
    print(f"Resolved provider_uri: {resolved_provider_uri}")
    patch_all(resolved_provider_uri)
    import qlib
    from qlib.constant import REG_CN

    qlib.init(provider_uri=resolved_provider_uri, region=REG_CN)

    resolved_ckpt_dir = (
        ckpt_dir
        or os.environ.get("CKPT_DIR", "")
        or "/kaggle/working/checkpoints"
    )
    resolved_tb_dir = (
        tb_dir
        or os.environ.get("TB_DIR", "")
        or "/kaggle/working/tb_log"
    )
    os.makedirs(resolved_ckpt_dir, exist_ok=True)
    os.makedirs(resolved_tb_dir, exist_ok=True)

    if device == "auto":
        resolved_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = device
    print(f"Resolved device: {resolved_device}")
    device = torch.device(resolved_device)

    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1

    train_start_time = '2014-01-01'
    train_end_time = '2018-12-31'
    valid_start_time = '2020-01-01'
    valid_end_time = '2020-12-31'
    test_start_time = '2021-01-01'
    test_end_time = '2022-12-31'
    stockdata_max_backtrack_days = 100
    stockdata_max_future_days = 30

    # You can re-implement AlphaCalculator instead of using QLibStockDataCalculator.
    data_train = StockData(instrument=market,
                           start_time=train_start_time,
                           end_time=train_end_time,
                           max_backtrack_days=stockdata_max_backtrack_days,
                           max_future_days=stockdata_max_future_days,
                           device=device)
    data_valid = StockData(instrument=market,
                           start_time=valid_start_time,
                           end_time=valid_end_time,
                           max_backtrack_days=stockdata_max_backtrack_days,
                           max_future_days=stockdata_max_future_days,
                           device=device)
    data_test = StockData(instrument=market,
                          start_time=test_start_time,
                          end_time=test_end_time,
                          max_backtrack_days=stockdata_max_backtrack_days,
                          max_future_days=stockdata_max_future_days,
                          device=device)
    calculator_train = QLibStockDataCalculator(data_train, target)
    calculator_valid = QLibStockDataCalculator(data_valid, target)
    calculator_test = QLibStockDataCalculator(data_test, target)

    pool = AlphaPool(
        capacity=pool_capacity,
        calculator=calculator_train,
        ic_lower_bound=None,
        l1_alpha=5e-3,
        reward_mode=reward_mode,
        re_mode=re_mode,
        lambda_ri=lambda_ri,
        ri_func_weight=ri_func_weight,
        ri_struct_weight=ri_struct_weight,
        ri_reg_weight=ri_reg_weight,
        ri_schedule_decay=ri_schedule_decay,
        ri_struct_value_bonus=ri_struct_value_bonus,
        ri_struct_underexplore_power=ri_struct_underexplore_power,
        ri_func_metric=ri_func_metric,
        ri_func_topk=ri_func_topk,
        ri_func_sample_size=ri_func_sample_size,
        ri_func_rankic_on_cpu=ri_func_rankic_on_cpu,
        ri_admission_gate=ri_admission_gate,
        ri_func_backend=ri_func_backend,
        ri_struct_backend=ri_struct_backend,
        ri_corr_threshold=ri_corr_threshold,
        ri_ast_similarity_threshold=ri_ast_similarity_threshold,
        ri_corr_value_bonus=ri_corr_value_bonus,
        ri_corr_underexplore_power=ri_corr_underexplore_power,
        ri_turnover_weight=ri_turnover_weight,
        ri_turnover_topk=ri_turnover_topk,
        ri_turnover_baseline=ri_turnover_baseline,
        ri_turnover_step_stride=ri_turnover_step_stride,
        profile_timing=profile_timing,
        optimize_every=optimize_every,
        optimize_n_iter=optimize_n_iter,
        ri_reg_l0=ri_reg_l0,
        ri_struct_topk=ri_struct_topk,
    )
    env = AlphaEnv(pool=pool, device=device, print_expr=False, reward_per_step=reward_per_step)

    if run_name:
        name_prefix = run_name
    else:
        step_tag = _format_step_tag(int(steps))
        name_prefix = f"b_seed{seed}_{backbone}_{reward_mode}_{step_tag}"
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    ckpt_run_dir = os.path.join(resolved_ckpt_dir, f"{name_prefix}_{timestamp}")
    tb_run_dir = os.path.join(resolved_tb_dir, f"{name_prefix}_{timestamp}")
    os.makedirs(ckpt_run_dir, exist_ok=True)
    os.makedirs(tb_run_dir, exist_ok=True)
    resolved_logdir = logdir or os.path.dirname(resolved_ckpt_dir)
    run_root_dir = os.path.join(resolved_logdir, f"{name_prefix}_{timestamp}")
    os.makedirs(run_root_dir, exist_ok=True)
    meta_path = os.path.join(run_root_dir, "run_meta.json")
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_id": f"{name_prefix}_{timestamp}",
                    "market": market,
                    "seed": seed,
                    "pool_capacity": pool_capacity,
                    "steps": steps,
                    "backbone": backbone,
                    "re_mode": re_mode,
                    "reward_mode": reward_mode,
                    "lambda_ri": lambda_ri,
                    "ri_func_weight": ri_func_weight,
                    "ri_struct_weight": ri_struct_weight,
                    "ri_reg_weight": ri_reg_weight,
                    "ri_schedule_decay": ri_schedule_decay,
                    "ri_struct_value_bonus": ri_struct_value_bonus,
                    "ri_struct_underexplore_power": ri_struct_underexplore_power,
                    "ri_func_metric": ri_func_metric,
                    "ri_func_topk": ri_func_topk,
                    "ri_func_sample_size": ri_func_sample_size,
                    "ri_func_rankic_on_cpu": ri_func_rankic_on_cpu,
                    "ri_admission_gate": ri_admission_gate,
                    "ri_func_backend": ri_func_backend,
                    "ri_struct_backend": ri_struct_backend,
                    "ri_corr_threshold": ri_corr_threshold,
                    "ri_ast_similarity_threshold": ri_ast_similarity_threshold,
                    "ri_corr_value_bonus": ri_corr_value_bonus,
                    "ri_corr_underexplore_power": ri_corr_underexplore_power,
                    "ri_turnover_weight": ri_turnover_weight,
                    "ri_turnover_topk": ri_turnover_topk,
                    "ri_turnover_baseline": ri_turnover_baseline,
                    "ri_turnover_step_stride": ri_turnover_step_stride,
                    "profile_timing": profile_timing,
                    "optimize_every": optimize_every,
                    "optimize_n_iter": optimize_n_iter,
                    "save_model_ckpt": save_model_ckpt,
                    "reward_per_step": reward_per_step,
                    "ri_reg_l0": ri_reg_l0,
                    "ri_struct_topk": ri_struct_topk,
                    "run_label": (
                        f"{market} | {backbone} | {reward_mode} | "
                        f"seed={seed} | pool={pool_capacity}"
                    ),
                    "timestamp": timestamp,
                    "ckpt_run_dir": ckpt_run_dir,
                    "tb_run_dir": tb_run_dir,
                    "run_root_dir": run_root_dir,
                    "provider_uri": resolved_provider_uri,
                    "train_start_time": train_start_time,
                    "train_end_time": train_end_time,
                    "train_start_year": int(train_start_time[:4]),
                    "train_end_year": int(train_end_time[:4]),
                    "valid_start_time": valid_start_time,
                    "valid_end_time": valid_end_time,
                    "test_start_time": test_start_time,
                    "test_end_time": test_end_time,
                    "stockdata_max_backtrack_days": stockdata_max_backtrack_days,
                    "stockdata_max_future_days": stockdata_max_future_days,
                    "target_expression": "Ref($close,-20)/$close-1",
                    "target_horizon_days": 20,
                    "data_windows": {
                        "train": {"start": train_start_time, "end": train_end_time},
                        "valid": {"start": valid_start_time, "end": valid_end_time},
                        "test": {"start": test_start_time, "end": test_end_time},
                    },
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    except Exception:
        pass

    checkpoint_callback = CustomCallback(
        save_freq=10000,
        show_freq=10000,
        save_path=ckpt_run_dir,
        run_dir=run_root_dir,
        valid_calculator=calculator_valid,
        test_calculator=calculator_test,
        save_model_ckpt=save_model_ckpt,
        name_prefix=name_prefix,
        timestamp=timestamp,
        verbose=verbose,
    )

    if backbone == "transformer":
        features_extractor_class = TransformerSharedNet
        features_extractor_kwargs = dict(
            n_encoder_layers=2,
            d_model=128,
            n_head=4,
            d_ffn=256,
            dropout=0.1,
            device=device,
        )
    else:
        features_extractor_class = LSTMSharedNet
        features_extractor_kwargs = dict(
            n_layers=2,
            d_model=128,
            dropout=0.1,
            device=device,
        )

    model = MaskablePPO(
        'MlpPolicy',
        env,
        policy_kwargs=dict(
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
        ),
        gamma=1.,
        ent_coef=0.01,
        batch_size=128,
        tensorboard_log=tb_run_dir,
        device=device,
        verbose=verbose,
    )
    model.learn(
        total_timesteps=steps,
        callback=checkpoint_callback,
        tb_log_name=f'{name_prefix}_{timestamp}',
    )




def _format_step_tag(steps: int) -> str:
    if steps % 1000 == 0:
        return f"{steps // 1000}k"
    return str(steps)


def _parse_seed_list(seed_value: Union[str, int]) -> List[int]:
    if isinstance(seed_value, int):
        return [seed_value]
    seed_str = str(seed_value)
    if "," in seed_str:
        return [int(x.strip()) for x in seed_str.split(",") if x.strip()]
    return [int(seed_str)]


def _resolve_steps(pool: int, step: Optional[int]) -> int:
    default_steps = {
        10: 250_000,
        20: 300_000,
        50: 350_000,
        100: 400_000
    }
    if step is not None:
        return int(step)
    return int(default_steps.get(int(pool), 200_000))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("seed_pos", nargs="?", type=str)
    parser.add_argument("code_pos", nargs="?", type=str)
    parser.add_argument("pool_pos", nargs="?", type=int)
    parser.add_argument("--seed", type=str, default=None)
    parser.add_argument("--code", type=str, default=None)
    parser.add_argument("--pool", type=int, default=None)
    parser.add_argument("--step", "--steps", dest="step", type=int, default=None)
    parser.add_argument("--backbone", type=str, default="lstm", choices=["lstm", "transformer"])
    parser.add_argument("--re_mode", type=str, default="ensemble", choices=["ensemble", "delta_best"])
    parser.add_argument("--reward_mode", type=str, default="re",
                        choices=["re", "re+func", "re+struct", "re+reg", "re+func+struct", "re+all",
                                 "re+func_v2", "re+struct_v2", "re+reg_v2",
                                 "re+struct_v2+reg_v2", "re+all_v2"])
    parser.add_argument("--lambda_ri", type=float, default=0.0)
    parser.add_argument("--ri_func_weight", type=float, default=1.0)
    parser.add_argument("--ri_struct_weight", type=float, default=1.0)
    parser.add_argument("--ri_reg_weight", type=float, default=1.0)
    parser.add_argument("--ri_schedule_decay", type=float, default=0.0)
    parser.add_argument("--ri_struct_value_bonus", type=float, default=0.1)
    parser.add_argument("--ri_struct_underexplore_power", type=float, default=1.0)
    parser.add_argument("--ri_func_metric", type=str, default="rankic", choices=["rankic", "ic"])
    parser.add_argument("--ri_func_topk", type=int, default=8)
    parser.add_argument("--ri_func_sample_size", type=int, default=128)
    parser.add_argument("--ri_func_rankic_on_cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ri_admission_gate", action="store_true")
    parser.add_argument("--ri_func_backend", type=str, default="auto",
                        choices=["auto", "mutual_ic", "residual", "corr_cluster"])
    parser.add_argument("--ri_struct_backend", type=str, default="token_bigram",
                        choices=["token_bigram", "ast_cluster"])
    parser.add_argument("--ri_corr_threshold", type=float, default=0.8)
    parser.add_argument("--ri_ast_similarity_threshold", type=float, default=0.9)
    parser.add_argument("--ri_corr_value_bonus", type=float, default=0.1)
    parser.add_argument("--ri_corr_underexplore_power", type=float, default=1.0)
    parser.add_argument("--ri_turnover_weight", type=float, default=0.0)
    parser.add_argument("--ri_turnover_topk", type=int, default=30)
    parser.add_argument("--ri_turnover_baseline", type=float, default=0.5)
    parser.add_argument("--ri_turnover_step_stride", type=int, default=5)
    parser.add_argument("--profile_timing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimize_every", type=int, default=2)
    parser.add_argument("--optimize_n_iter", type=int, default=256)
    parser.add_argument("--save_model_ckpt", action="store_true")
    parser.add_argument("--reward_per_step", type=float, default=REWARD_PER_STEP)
    parser.add_argument("--ri_reg_l0", type=float, default=None)
    parser.add_argument("--ri_struct_topk", type=int, default=5)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--logdir", type=str, default="")
    parser.add_argument("--provider_uri", type=str, default="")
    parser.add_argument("--ckpt_dir", type=str, default="")
    parser.add_argument("--tb_dir", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--verbose", type=int, default=0)
    return parser


if __name__ == '__main__':
    parser = _build_arg_parser()
    args = parser.parse_args()

    seed_arg = args.seed if args.seed is not None else args.seed_pos
    code_arg = args.code if args.code is not None else args.code_pos
    pool_arg = args.pool if args.pool is not None else args.pool_pos
    if seed_arg is None or code_arg is None or pool_arg is None:
        parser.error("seed, code, pool are required (positional or via --seed/--code/--pool).")

    steps = _resolve_steps(pool_arg, args.step)
    for seed in _parse_seed_list(seed_arg):
        main(
            seed=seed,
            market=code_arg,
            pool_capacity=int(pool_arg),
            steps=steps,
            backbone=args.backbone,
            re_mode=args.re_mode,
            reward_mode=args.reward_mode,
            lambda_ri=args.lambda_ri,
            ri_func_weight=args.ri_func_weight,
            ri_struct_weight=args.ri_struct_weight,
            ri_reg_weight=args.ri_reg_weight,
            ri_schedule_decay=args.ri_schedule_decay,
            ri_struct_value_bonus=args.ri_struct_value_bonus,
            ri_struct_underexplore_power=args.ri_struct_underexplore_power,
            ri_func_metric=args.ri_func_metric,
            ri_func_topk=args.ri_func_topk,
            ri_func_sample_size=args.ri_func_sample_size,
            ri_func_rankic_on_cpu=args.ri_func_rankic_on_cpu,
            ri_admission_gate=args.ri_admission_gate,
            ri_func_backend=args.ri_func_backend,
            ri_struct_backend=args.ri_struct_backend,
            ri_corr_threshold=args.ri_corr_threshold,
            ri_ast_similarity_threshold=args.ri_ast_similarity_threshold,
            ri_corr_value_bonus=args.ri_corr_value_bonus,
            ri_corr_underexplore_power=args.ri_corr_underexplore_power,
            ri_turnover_weight=args.ri_turnover_weight,
            ri_turnover_topk=args.ri_turnover_topk,
            ri_turnover_baseline=args.ri_turnover_baseline,
            ri_turnover_step_stride=args.ri_turnover_step_stride,
            profile_timing=args.profile_timing,
            optimize_every=args.optimize_every,
            optimize_n_iter=args.optimize_n_iter,
            save_model_ckpt=args.save_model_ckpt,
            reward_per_step=args.reward_per_step,
            ri_reg_l0=args.ri_reg_l0,
            ri_struct_topk=args.ri_struct_topk,
            run_name=args.run_name,
            logdir=args.logdir,
            provider_uri=args.provider_uri,
            ckpt_dir=args.ckpt_dir,
            tb_dir=args.tb_dir,
            device=args.device,
            verbose=args.verbose,
        )
