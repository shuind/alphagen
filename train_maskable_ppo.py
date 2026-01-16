import csv
import json
import math
import os
from typing import Optional, Tuple, Union
from datetime import datetime
import fire

import numpy as np
import torch
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from alphagen.data.calculator import AlphaCalculator

from alphagen.data.expression import *
from alphagen.models.alpha_pool import AlphaPool, AlphaPoolBase
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils.random import reseed_everything
from alphagen.rl.env.core import AlphaEnvCore
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.compat import patch_all


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
                "best_rankic": float(rank_ic_test) if rank_ic_test is not None else math.nan,
                "mean_ic": mean_ic,
                "mean_rankic": float(rank_ic_test) if rank_ic_test is not None else math.nan,
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
        l1_alpha=5e-3
    )
    env = AlphaEnv(pool=pool, device=device, print_expr=True)

    name_prefix = f"new_{market}_{pool_capacity}_{seed}"
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    ckpt_run_dir = os.path.join(resolved_ckpt_dir, f"{name_prefix}_{timestamp}")
    tb_run_dir = os.path.join(resolved_tb_dir, f"{name_prefix}_{timestamp}")
    os.makedirs(ckpt_run_dir, exist_ok=True)
    os.makedirs(tb_run_dir, exist_ok=True)
    run_root_dir = os.path.join(os.path.dirname(resolved_ckpt_dir), f"{name_prefix}_{timestamp}")
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

    model = MaskablePPO(
        'MlpPolicy',
        env,
        policy_kwargs=dict(
            features_extractor_class=LSTMSharedNet,
            features_extractor_kwargs=dict(
                n_layers=2,
                d_model=128,
                dropout=0.1,
                device=device,
            ),
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


def fire_helper(
    seed: Union[int, Tuple[int]],
    code: str,
    pool: int,
    step: int = None
):
    if isinstance(seed, int):
        seed = (seed, )
    default_steps = {
        10: 250_000,
        20: 300_000,
        50: 350_000,
        100: 400_000
    }
    for _seed in seed:
        main(_seed,
             code,
             pool,
             default_steps[int(pool)] if step is None else int(step)
             )


if __name__ == '__main__':
    fire.Fire(fire_helper)
