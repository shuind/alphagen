import argparse
import json
import os
from collections import Counter
from datetime import datetime
from typing import Optional

import torch
from sb3_contrib.ppo_mask import MaskablePPO

from alphagen.config import REWARD_PER_STEP
from alphagen.data.expression import Feature, Ref
from alphagen.models.alpha_pool import AlphaPoolBase
from alphagen.utils.random import reseed_everything
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.compat import patch_all
from alphagen_qlib.stock_data import FeatureType, StockData
from new.classic_factors import load_classic_factors
from new.env import MultiHeadAlphaEnv
from new.head_pretrain import pretrain_policy_heads
from new.policy import MultiHeadMaskablePolicy, MultiHeadTransformerFeatures
from new.pool import MultiHeadAlphaPool
from train_maskable_ppo import CustomCallback


def _format_step_tag(steps: int) -> str:
    if steps % 1000 == 0:
        return f"{steps // 1000}k"
    return str(steps)


def _resolve_steps(pool: int, step: Optional[int]) -> int:
    if step is not None:
        return int(step)
    default_steps = {10: 64_000, 20: 64_000, 50: 200_000}
    return int(default_steps.get(int(pool), 64_000))


def main(
    seed: int = 0,
    market: str = "tcsi300",
    pool_capacity: int = 10,
    steps: int = 64_000,
    method: str = "multihead_intrinsic",
    intrinsic_beta: float = 0.1,
    simple_bias: float = 0.4,
    ts_bias: float = 0.5,
    optimize_every: int = 2,
    optimize_n_iter: int = 256,
    head_pretrain_epochs: int = 20,
    head_pretrain_lr: float = 1e-3,
    head_pretrain_batch_size: int = 128,
    classic_factor_csv: str = "",
    no_head_pretrain: bool = False,
    save_pretrain_ckpt: bool = False,
    pretrain_ckpt_path: str = "",
    load_pretrain_ckpt: str = "",
    save_model_ckpt: bool = False,
    reward_per_step: float = REWARD_PER_STEP,
    run_name: str = "",
    logdir: str = "",
    provider_uri: str = "",
    ckpt_dir: str = "",
    tb_dir: str = "",
    device: str = "auto",
    verbose: int = 0,
) -> None:
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

    resolved_device = "cuda:0" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    print(f"Resolved device: {resolved_device}")
    torch_device = torch.device(resolved_device)

    resolved_ckpt_dir = ckpt_dir or os.environ.get("CKPT_DIR", "") or "/kaggle/working/checkpoints"
    resolved_tb_dir = tb_dir or os.environ.get("TB_DIR", "") or "/kaggle/working/tb_log"
    os.makedirs(resolved_ckpt_dir, exist_ok=True)
    os.makedirs(resolved_tb_dir, exist_ok=True)

    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1
    train_start_time = "2014-01-01"
    train_end_time = "2018-12-31"
    valid_start_time = "2020-01-01"
    valid_end_time = "2020-12-31"
    test_start_time = "2021-01-01"
    test_end_time = "2022-12-31"
    max_backtrack_days = 100
    max_future_days = 30

    data_train = StockData(
        instrument=market,
        start_time=train_start_time,
        end_time=train_end_time,
        max_backtrack_days=max_backtrack_days,
        max_future_days=max_future_days,
        device=torch_device,
    )
    data_valid = StockData(
        instrument=market,
        start_time=valid_start_time,
        end_time=valid_end_time,
        max_backtrack_days=max_backtrack_days,
        max_future_days=max_future_days,
        device=torch_device,
    )
    data_test = StockData(
        instrument=market,
        start_time=test_start_time,
        end_time=test_end_time,
        max_backtrack_days=max_backtrack_days,
        max_future_days=max_future_days,
        device=torch_device,
    )
    calculator_train = QLibStockDataCalculator(data_train, target)
    calculator_valid = QLibStockDataCalculator(data_valid, target)
    calculator_test = QLibStockDataCalculator(data_test, target)

    beta = float(intrinsic_beta) if method == "multihead_intrinsic" else 0.0
    pool: AlphaPoolBase = MultiHeadAlphaPool(
        capacity=pool_capacity,
        calculator=calculator_train,
        ic_lower_bound=None,
        l1_alpha=5e-3,
        re_mode="ensemble",
        intrinsic_beta=beta,
        profile_timing=True,
        optimize_every=optimize_every,
        optimize_n_iter=optimize_n_iter,
        device=torch_device,
    )
    env = MultiHeadAlphaEnv(
        pool=pool,
        method=method,
        device=torch_device,
        print_expr=False,
        reward_per_step=reward_per_step,
    )

    step_tag = _format_step_tag(int(steps))
    name_prefix = run_name or f"mh_seed{seed}_{method}_p{pool_capacity}_{step_tag}"
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_id = f"{name_prefix}_{timestamp}"
    ckpt_run_dir = os.path.join(resolved_ckpt_dir, run_id)
    tb_run_dir = os.path.join(resolved_tb_dir, run_id)
    resolved_logdir = logdir or os.path.dirname(resolved_ckpt_dir)
    run_root_dir = os.path.join(resolved_logdir, run_id)
    os.makedirs(ckpt_run_dir, exist_ok=True)
    os.makedirs(tb_run_dir, exist_ok=True)
    os.makedirs(run_root_dir, exist_ok=True)

    if load_pretrain_ckpt and not os.path.exists(load_pretrain_ckpt):
        raise FileNotFoundError(f"pretrain checkpoint not found: {load_pretrain_ckpt}")

    head_pretrain_enabled = (
        method != "single_transformer"
        and not bool(no_head_pretrain)
        and int(head_pretrain_epochs) > 0
        and not bool(load_pretrain_ckpt)
    )
    classic_factors = []
    skipped_classic_factors = []
    if method != "single_transformer" and not bool(no_head_pretrain) and not bool(load_pretrain_ckpt):
        classic_factors, skipped_classic_factors = load_classic_factors(classic_factor_csv)
        print(
            "[classic-factors] "
            f"loaded={len(classic_factors)} skipped={len(skipped_classic_factors)} "
            f"heads={dict(Counter(f.head for f in classic_factors))}"
        )

    run_meta = {
        "run_id": run_id,
        "market": market,
        "seed": seed,
        "pool_capacity": pool_capacity,
        "steps": steps,
        "method": method,
        "backbone": "shared_transformer",
        "reward_mode": "re",
        "intrinsic_beta": beta,
        "simple_bias": simple_bias,
        "ts_bias": ts_bias,
        "optimize_every": optimize_every,
        "optimize_n_iter": optimize_n_iter,
        "head_pretrain_enabled": head_pretrain_enabled,
        "head_pretrain_epochs": int(head_pretrain_epochs),
        "head_pretrain_lr": float(head_pretrain_lr),
        "head_pretrain_batch_size": int(head_pretrain_batch_size),
        "classic_factor_csv": classic_factor_csv,
        "save_pretrain_ckpt": bool(save_pretrain_ckpt),
        "pretrain_ckpt_path": pretrain_ckpt_path,
        "load_pretrain_ckpt": load_pretrain_ckpt,
        "classic_factor_count": len(classic_factors),
        "classic_factor_skipped_count": len(skipped_classic_factors),
        "classic_factor_head_counts": dict(Counter(f.head for f in classic_factors)),
        "classic_factor_skipped": [s.__dict__ for s in skipped_classic_factors[:20]],
        "provider_uri": resolved_provider_uri,
        "train_start_time": train_start_time,
        "train_end_time": train_end_time,
        "train_start_year": int(train_start_time[:4]),
        "train_end_year": int(train_end_time[:4]),
        "valid_start_time": valid_start_time,
        "valid_end_time": valid_end_time,
        "test_start_time": test_start_time,
        "test_end_time": test_end_time,
        "stockdata_max_backtrack_days": max_backtrack_days,
        "stockdata_max_future_days": max_future_days,
        "target_expression": "Ref($close,-20)/$close-1",
        "ckpt_run_dir": ckpt_run_dir,
        "tb_run_dir": tb_run_dir,
        "run_root_dir": run_root_dir,
        "timestamp": timestamp,
    }

    callback = CustomCallback(
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

    model = MaskablePPO(
        MultiHeadMaskablePolicy,
        env,
        policy_kwargs=dict(
            features_extractor_class=MultiHeadTransformerFeatures,
            features_extractor_kwargs=dict(
                n_encoder_layers=2,
                d_model=128,
                n_head=4,
                d_ffn=256,
                dropout=0.1,
                device=torch_device,
            ),
            simple_bias=simple_bias,
            ts_bias=ts_bias,
        ),
        gamma=1.0,
        ent_coef=0.01,
        batch_size=128,
        tensorboard_log=tb_run_dir,
        device=torch_device,
        verbose=verbose,
    )

    if load_pretrain_ckpt:
        print(f"[head-pretrain] loading checkpoint: {load_pretrain_ckpt}")
        loaded_model = MaskablePPO.load(
            load_pretrain_ckpt,
            env=env,
            device=torch_device,
            tensorboard_log=tb_run_dir,
        )
        model.policy.load_state_dict(loaded_model.policy.state_dict())
        pretrain_summary = {
            "enabled": False,
            "loaded": True,
            "loaded_path": load_pretrain_ckpt,
            "epochs": 0,
            "factor_count": 0,
            "sample_count": 0,
            "loss_start": None,
            "loss_end": None,
            "losses": [],
        }
    elif head_pretrain_enabled:
        pretrain_summary = pretrain_policy_heads(
            model=model,
            factors=classic_factors,
            epochs=int(head_pretrain_epochs),
            lr=float(head_pretrain_lr),
            batch_size=int(head_pretrain_batch_size),
            seed=int(seed),
        )
    else:
        pretrain_summary = {
            "enabled": False,
            "epochs": int(head_pretrain_epochs),
            "factor_count": len(classic_factors),
            "sample_count": 0,
            "loss_start": None,
            "loss_end": None,
            "losses": [],
        }
    if save_pretrain_ckpt:
        resolved_pretrain_ckpt_path = pretrain_ckpt_path or os.path.join(ckpt_run_dir, "pretrained_policy.zip")
        pretrain_parent = os.path.dirname(resolved_pretrain_ckpt_path)
        if pretrain_parent:
            os.makedirs(pretrain_parent, exist_ok=True)
        model.save(resolved_pretrain_ckpt_path)
        pretrain_summary["saved"] = True
        pretrain_summary["saved_path"] = resolved_pretrain_ckpt_path
        run_meta["pretrain_ckpt_path"] = resolved_pretrain_ckpt_path
        print(f"[head-pretrain] saved checkpoint: {resolved_pretrain_ckpt_path}")
    run_meta["head_pretrain_summary"] = pretrain_summary
    with open(os.path.join(run_root_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)
    model.learn(total_timesteps=steps, callback=callback, tb_log_name=run_id)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train clean shared-Transformer multi-head AlphaGen PPO.")
    parser.add_argument("seed_pos", nargs="?", type=str)
    parser.add_argument("market_pos", nargs="?", type=str)
    parser.add_argument("pool_pos", nargs="?", type=int)
    parser.add_argument("--seed", type=str, default=None)
    parser.add_argument("--market", type=str, default=None)
    parser.add_argument("--pool", type=int, default=None)
    parser.add_argument("--step", "--steps", dest="step", type=int, default=None)
    parser.add_argument("--method", choices=["single_transformer", "multihead", "multihead_intrinsic"], default="multihead_intrinsic")
    parser.add_argument("--intrinsic-beta", type=float, default=0.1)
    parser.add_argument("--simple-bias", type=float, default=0.4)
    parser.add_argument("--ts-bias", type=float, default=0.5)
    parser.add_argument("--optimize_every", type=int, default=2)
    parser.add_argument("--optimize_n_iter", type=int, default=256)
    parser.add_argument("--head-pretrain-epochs", type=int, default=20)
    parser.add_argument("--head-pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--head-pretrain-batch-size", type=int, default=128)
    parser.add_argument("--classic-factor-csv", type=str, default="")
    parser.add_argument("--no-head-pretrain", action="store_true")
    parser.add_argument("--save-pretrain-ckpt", action="store_true")
    parser.add_argument("--pretrain-ckpt-path", type=str, default="")
    parser.add_argument("--load-pretrain-ckpt", type=str, default="")
    parser.add_argument("--save_model_ckpt", action="store_true")
    parser.add_argument("--reward_per_step", type=float, default=REWARD_PER_STEP)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--logdir", type=str, default="")
    parser.add_argument("--provider_uri", type=str, default="")
    parser.add_argument("--ckpt_dir", type=str, default="")
    parser.add_argument("--tb_dir", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--verbose", type=int, default=0)
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()
    seed_arg = args.seed if args.seed is not None else args.seed_pos
    market_arg = args.market if args.market is not None else args.market_pos
    pool_arg = args.pool if args.pool is not None else args.pool_pos
    if seed_arg is None or market_arg is None or pool_arg is None:
        parser.error("seed, market, pool are required.")
    steps = _resolve_steps(int(pool_arg), args.step)
    main(
        seed=int(seed_arg),
        market=str(market_arg),
        pool_capacity=int(pool_arg),
        steps=steps,
        method=args.method,
        intrinsic_beta=args.intrinsic_beta,
        simple_bias=args.simple_bias,
        ts_bias=args.ts_bias,
        optimize_every=args.optimize_every,
        optimize_n_iter=args.optimize_n_iter,
        head_pretrain_epochs=args.head_pretrain_epochs,
        head_pretrain_lr=args.head_pretrain_lr,
        head_pretrain_batch_size=args.head_pretrain_batch_size,
        classic_factor_csv=args.classic_factor_csv,
        no_head_pretrain=args.no_head_pretrain,
        save_pretrain_ckpt=args.save_pretrain_ckpt,
        pretrain_ckpt_path=args.pretrain_ckpt_path,
        load_pretrain_ckpt=args.load_pretrain_ckpt,
        save_model_ckpt=args.save_model_ckpt,
        reward_per_step=args.reward_per_step,
        run_name=args.run_name,
        logdir=args.logdir,
        provider_uri=args.provider_uri,
        ckpt_dir=args.ckpt_dir,
        tb_dir=args.tb_dir,
        device=args.device,
        verbose=args.verbose,
    )
