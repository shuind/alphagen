import argparse
import csv
import json
import math
import os
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
        "mean_rankic",
        "test_rankic",
        "reward_total",
        "RE",
        "RI_func",
        "RI_struct",
        "RI_reg",
        "reward_lambda_t",
        "cluster_id",
        "cluster_count",
        "cluster_mean_re",
        "cluster_positive_re_rate",
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
        self.logger.record('pool/size', self.pool.size)
        self.logger.record('pool/significant', (np.abs(self.pool.weights[:self.pool.size]) > 1e-4).sum())
        self.logger.record('pool/best_ic_ret', self.pool.best_ic_ret)
        self.logger.record('pool/eval_cnt', self.pool.eval_cnt)
        ic_test, rank_ic_test = self.pool.test_ensemble(self.test_calculator)
        self.logger.record('test/ic', ic_test)
        self.logger.record('test/rank_ic', rank_ic_test)
        rank_ic_test_value = float(rank_ic_test) if rank_ic_test is not None else math.nan
        pool_size = getattr(self.pool, "size", math.nan)
        if isinstance(pool_size, (int, np.integer)) and pool_size > 0 and hasattr(self.pool, "single_ics"):
            mean_ic = float(np.nanmean(self.pool.single_ics[:pool_size]))
        else:
            mean_ic = math.nan
        log_metrics_csv(
            self.run_dir or self.save_path,
            self.num_timesteps,
            {
                "pool_size": pool_size,
                "best_ic": getattr(self.pool, "best_ic_ret", math.nan),
                "best_rankic": rank_ic_test_value,
                "mean_ic": mean_ic,
                "mean_rankic": math.nan,
                "test_rankic": rank_ic_test_value,
                "reward_total": getattr(self.pool, "last_reward_info", {}).get("reward_total", math.nan),
                "RE": getattr(self.pool, "last_reward_info", {}).get("re", math.nan),
                "RI_func": getattr(self.pool, "last_reward_info", {}).get("ri_func", math.nan),
                "RI_struct": getattr(self.pool, "last_reward_info", {}).get("ri_struct", math.nan),
                "RI_reg": getattr(self.pool, "last_reward_info", {}).get("ri_reg", math.nan),
                "reward_lambda_t": getattr(self.pool, "last_reward_info", {}).get("reward_lambda_t", math.nan),
                "cluster_id": getattr(self.pool, "last_reward_info", {}).get("cluster_id", math.nan),
                "cluster_count": getattr(self.pool, "last_reward_info", {}).get("cluster_count", math.nan),
                "cluster_mean_re": getattr(self.pool, "last_reward_info", {}).get("cluster_mean_re", math.nan),
                "cluster_positive_re_rate": getattr(self.pool, "last_reward_info", {}).get("cluster_positive_re_rate", math.nan),
            },
        )
        self.save_checkpoint()

    def save_checkpoint(self):
        path = os.path.join(self.save_path, f'{self.num_timesteps}_steps')
        self.model.save(path)   # type: ignore
        if self.verbose > 1:
            print(f'Saving model checkpoint to {path}')
        with open(f'{path}_pool.json', 'w') as f:
            json.dump(self.pool.to_dict(), f)

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
    ri_admission_gate: bool = False,
    reward_per_step: float = REWARD_PER_STEP,
    ri_reg_l0: Optional[float] = None,
    ri_struct_topk: int = 5,
    run_name: str = "",
    logdir: str = "",
    provider_uri: str = "",
    ckpt_dir: str = "",
    tb_dir: str = "",
    device: str = "auto",
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

    # You can re-implement AlphaCalculator instead of using QLibStockDataCalculator.
    data_train = StockData(instrument=market,
                           start_time='2010-01-01',
                           end_time='2019-12-31')
    data_valid = StockData(instrument=market,
                           start_time='2020-01-01',
                           end_time='2020-12-31')
    data_test = StockData(instrument=market,
                          start_time='2021-01-01',
                          end_time='2022-12-31')
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
        ri_admission_gate=ri_admission_gate,
        ri_reg_l0=ri_reg_l0,
        ri_struct_topk=ri_struct_topk,
    )
    env = AlphaEnv(pool=pool, device=device, print_expr=True, reward_per_step=reward_per_step)

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
                    "ri_admission_gate": ri_admission_gate,
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
        name_prefix=name_prefix,
        timestamp=timestamp,
        verbose=1,
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
        verbose=1,
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
                                 "re_v2", "re_v2+func", "re_v2+struct", "re_v2+reg", "re_v2+all"])
    parser.add_argument("--lambda_ri", type=float, default=0.0)
    parser.add_argument("--ri_func_weight", type=float, default=1.0)
    parser.add_argument("--ri_struct_weight", type=float, default=1.0)
    parser.add_argument("--ri_reg_weight", type=float, default=1.0)
    parser.add_argument("--ri_schedule_decay", type=float, default=0.0)
    parser.add_argument("--ri_struct_value_bonus", type=float, default=0.1)
    parser.add_argument("--ri_struct_underexplore_power", type=float, default=1.0)
    parser.add_argument("--ri_func_metric", type=str, default="rankic", choices=["rankic", "ic"])
    parser.add_argument("--ri_admission_gate", action="store_true")
    parser.add_argument("--reward_per_step", type=float, default=REWARD_PER_STEP)
    parser.add_argument("--ri_reg_l0", type=float, default=None)
    parser.add_argument("--ri_struct_topk", type=int, default=5)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--logdir", type=str, default="")
    parser.add_argument("--provider_uri", type=str, default="")
    parser.add_argument("--ckpt_dir", type=str, default="")
    parser.add_argument("--tb_dir", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
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
            ri_admission_gate=args.ri_admission_gate,
            reward_per_step=args.reward_per_step,
            ri_reg_l0=args.ri_reg_l0,
            ri_struct_topk=args.ri_struct_topk,
            run_name=args.run_name,
            logdir=args.logdir,
            provider_uri=args.provider_uri,
            ckpt_dir=args.ckpt_dir,
            tb_dir=args.tb_dir,
            device=args.device,
        )
