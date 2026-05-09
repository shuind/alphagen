import argparse
import json
import os
from collections import Counter
from datetime import datetime
from typing import List, Optional

import torch
from sb3_contrib.ppo_mask import MaskablePPO

from alphagen.config import REWARD_PER_STEP
from alphagen.data.expression import Feature, Ref
from alphagen.models.alpha_pool import AlphaPoolBase
from alphagen.utils.random import reseed_everything
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.compat import patch_all
from alphagen_qlib.stock_data import FeatureType, StockData
from new.classic_factors import DEFAULT_LOSS_WEIGHTS, load_classic_factor_bank
from new.env import HEAD_NAMES, MultiHeadAlphaEnv
from new.head_pretrain import parse_loss_weights, pretrain_policy_heads
from new.motif_bank import motif_summary
from new.motif_env import MotifEditAlphaEnv, N_ACTIONS as MOTIF_EDIT_ACTIONS
from new.motif_pool import MotifEditAlphaPool
from new.policy import MultiHeadMaskablePolicy, MultiHeadTransformerFeatures
from new.pool import MultiHeadAlphaPool
from new.typed_qd import TypedQDAlphaPool
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


def _parse_years(text: str, default_start: int, default_end: int) -> List[int]:
    if not text:
        return list(range(int(default_start), int(default_end) + 1))
    years: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lhs, rhs = part.split("-", 1)
            years.extend(range(int(lhs), int(rhs) + 1))
        else:
            years.append(int(part))
    return sorted(dict.fromkeys(years))


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
    classic_factor_bank: str = "strong",
    classic_factor_augment: bool = True,
    pretrain_loss_weights: str = DEFAULT_LOSS_WEIGHTS,
    no_pretrain_aux_loss: bool = False,
    no_head_pretrain: bool = False,
    save_pretrain_ckpt: bool = False,
    pretrain_ckpt_path: str = "",
    load_pretrain_ckpt: str = "",
    motif_max_edits: int = 4,
    motif_prior_eta: float = 0.02,
    behavior_novelty_beta: float = 0.05,
    behavior_archive_size: int = 128,
    typed_robust_years: str = "2014,2015,2016,2017,2018",
    typed_robust_lambda: float = 0.5,
    typed_robust_bottom_k: int = 0,
    qd_cell_capacity: int = 2,
    qd_behavior_threshold: float = 0.7,
    qd_bonus: float = 0.02,
    typed_min_robust_score: float = -1.0,
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
    valid_start_time = "2019-01-01"
    valid_end_time = "2019-12-31"
    test_start_time = "2019-01-01"
    test_end_time = "2019-12-31"
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

    token_methods = {"single_transformer", "multihead", "multihead_intrinsic"}
    typed_methods = {"typed_qd", "typed_qd_intrinsic"}
    token_multihead_methods = {"multihead", "multihead_intrinsic"} | typed_methods
    motif_methods = {"motif_edit", "motif_edit_intrinsic"}
    if method not in token_methods | motif_methods | typed_methods:
        raise ValueError(f"unsupported method: {method}")

    robust_years: List[int] = []
    robust_calculators = []
    if method in typed_methods:
        robust_years = _parse_years(typed_robust_years, int(train_start_time[:4]), int(train_end_time[:4]))
        for year in robust_years:
            yearly_data = StockData(
                instrument=market,
                start_time=f"{year}-01-01",
                end_time=f"{year}-12-31",
                max_backtrack_days=max_backtrack_days,
                max_future_days=max_future_days,
                device=torch_device,
            )
            robust_calculators.append(QLibStockDataCalculator(yearly_data, target))
        print(
            "[typed-qd] "
            f"robust_years={robust_years} lambda={float(typed_robust_lambda)} "
            f"bottom_k={int(typed_robust_bottom_k)} cell_capacity={int(qd_cell_capacity)}"
        )

    if method in typed_methods:
        beta = float(qd_bonus) if method == "typed_qd_intrinsic" else 0.0
        pool: AlphaPoolBase = TypedQDAlphaPool(
            capacity=pool_capacity,
            calculator=calculator_train,
            ic_lower_bound=None,
            l1_alpha=5e-3,
            re_mode="ensemble",
            robust_calculators=robust_calculators,
            robust_lambda=float(typed_robust_lambda),
            robust_bottom_k=int(typed_robust_bottom_k),
            qd_cell_capacity=int(qd_cell_capacity),
            qd_behavior_threshold=float(qd_behavior_threshold),
            qd_bonus=beta,
            min_robust_score=float(typed_min_robust_score),
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
    elif method in motif_methods:
        beta = float(behavior_novelty_beta) if method == "motif_edit_intrinsic" else 0.0
        pool = MotifEditAlphaPool(
            capacity=pool_capacity,
            calculator=calculator_train,
            ic_lower_bound=None,
            l1_alpha=5e-3,
            re_mode="ensemble",
            behavior_novelty_beta=beta,
            motif_prior_eta=float(motif_prior_eta),
            behavior_archive_size=int(behavior_archive_size),
            profile_timing=True,
            optimize_every=optimize_every,
            optimize_n_iter=optimize_n_iter,
            device=torch_device,
        )
        env = MotifEditAlphaEnv(
            pool=pool,
            method=method,
            max_edits=int(motif_max_edits),
            reward_per_step=reward_per_step,
        )
    else:
        beta = float(intrinsic_beta) if method == "multihead_intrinsic" else 0.0
        pool = MultiHeadAlphaPool(
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
        method in token_multihead_methods
        and not bool(no_head_pretrain)
        and int(head_pretrain_epochs) > 0
        and not bool(load_pretrain_ckpt)
    )
    classic_factors = []
    skipped_classic_factors = []
    classic_factor_stats = {}
    if method in token_multihead_methods and not bool(no_head_pretrain) and not bool(load_pretrain_ckpt):
        factor_bank_result = load_classic_factor_bank(
            csv_path=classic_factor_csv,
            bank=classic_factor_bank,
            augment=bool(classic_factor_augment),
        )
        classic_factors = factor_bank_result.factors
        skipped_classic_factors = factor_bank_result.skipped
        classic_factor_stats = factor_bank_result.stats
        print(
            "[classic-factors] "
            f"bank={classic_factor_bank} augment={bool(classic_factor_augment)} "
            f"loaded={len(classic_factors)} skipped={len(skipped_classic_factors)} "
            f"deduped={classic_factor_stats.get('deduped_count', 0)} "
            f"heads={dict(Counter(f.head for f in classic_factors))}"
        )
    parsed_pretrain_loss_weights = parse_loss_weights(pretrain_loss_weights)

    run_meta = {
        "run_id": run_id,
        "market": market,
        "seed": seed,
        "pool_capacity": pool_capacity,
        "steps": steps,
        "method": method,
        "backbone": "motif_edit_mlp" if method in motif_methods else "shared_transformer",
        "head_names": list(HEAD_NAMES),
        "reward_mode": "re",
        "intrinsic_beta": beta,
        "motif_method": bool(method in motif_methods),
        "motif_max_edits": int(motif_max_edits),
        "motif_prior_eta": float(motif_prior_eta),
        "behavior_novelty_beta": float(beta if method in motif_methods else 0.0),
        "behavior_archive_size": int(behavior_archive_size),
        "motif_summary": motif_summary() if method in motif_methods else {},
        "motif_edit_action_count": int(MOTIF_EDIT_ACTIONS) if method in motif_methods else 0,
        "typed_qd_method": bool(method in typed_methods),
        "typed_robust_years": robust_years,
        "typed_robust_lambda": float(typed_robust_lambda),
        "typed_robust_bottom_k": int(typed_robust_bottom_k),
        "qd_cell_capacity": int(qd_cell_capacity),
        "qd_behavior_threshold": float(qd_behavior_threshold),
        "qd_bonus": float(beta if method in typed_methods else 0.0),
        "typed_min_robust_score": float(typed_min_robust_score),
        "simple_bias": simple_bias,
        "ts_bias": ts_bias,
        "optimize_every": optimize_every,
        "optimize_n_iter": optimize_n_iter,
        "head_pretrain_enabled": head_pretrain_enabled,
        "head_pretrain_epochs": int(head_pretrain_epochs),
        "head_pretrain_lr": float(head_pretrain_lr),
        "head_pretrain_batch_size": int(head_pretrain_batch_size),
        "classic_factor_csv": classic_factor_csv,
        "classic_factor_bank": classic_factor_bank,
        "classic_factor_augment": bool(classic_factor_augment),
        "pretrain_loss_weights": parsed_pretrain_loss_weights,
        "pretrain_aux_loss_enabled": not bool(no_pretrain_aux_loss),
        "save_pretrain_ckpt": bool(save_pretrain_ckpt),
        "pretrain_ckpt_path": pretrain_ckpt_path,
        "load_pretrain_ckpt": load_pretrain_ckpt,
        "classic_factor_count": len(classic_factors),
        "classic_factor_skipped_count": len(skipped_classic_factors),
        "classic_factor_head_counts": dict(Counter(f.head for f in classic_factors)),
        "classic_factor_family_counts": dict(Counter((f.family or f.head) for f in classic_factors)),
        "classic_factor_source_counts": dict(Counter(f.source for f in classic_factors)),
        "classic_factor_stats": classic_factor_stats,
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

    if method in motif_methods:
        motif_n_steps = max(64, min(2048, int(steps)))
        model = MaskablePPO(
            "MlpPolicy",
            env,
            gamma=1.0,
            ent_coef=0.01,
            n_steps=motif_n_steps,
            batch_size=min(128, motif_n_steps),
            tensorboard_log=tb_run_dir,
            device=torch_device,
            verbose=verbose,
        )
    else:
        token_n_steps = max(64, min(2048, int(steps)))
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
            n_steps=token_n_steps,
            batch_size=min(128, token_n_steps),
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
            "next_losses": [],
            "head_losses": [],
            "attr_losses": [],
            "aux_loss_enabled": False,
            "loss_weights": parsed_pretrain_loss_weights,
        }
    elif head_pretrain_enabled:
        pretrain_summary = pretrain_policy_heads(
            model=model,
            factors=classic_factors,
            epochs=int(head_pretrain_epochs),
            lr=float(head_pretrain_lr),
            batch_size=int(head_pretrain_batch_size),
            seed=int(seed),
            loss_weights=parsed_pretrain_loss_weights,
            use_aux_loss=not bool(no_pretrain_aux_loss),
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
            "next_losses": [],
            "head_losses": [],
            "attr_losses": [],
            "aux_loss_enabled": not bool(no_pretrain_aux_loss),
            "loss_weights": parsed_pretrain_loss_weights,
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
    parser.add_argument(
        "--method",
        choices=[
            "single_transformer",
            "multihead",
            "multihead_intrinsic",
            "motif_edit",
            "motif_edit_intrinsic",
            "typed_qd",
            "typed_qd_intrinsic",
        ],
        default="multihead_intrinsic",
    )
    parser.add_argument("--intrinsic-beta", type=float, default=0.1)
    parser.add_argument("--simple-bias", type=float, default=0.4)
    parser.add_argument("--ts-bias", type=float, default=0.5)
    parser.add_argument("--optimize_every", type=int, default=2)
    parser.add_argument("--optimize_n_iter", type=int, default=256)
    parser.add_argument("--head-pretrain-epochs", type=int, default=20)
    parser.add_argument("--head-pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--head-pretrain-batch-size", type=int, default=128)
    parser.add_argument("--classic-factor-csv", type=str, default="")
    parser.add_argument("--classic-factor-bank", choices=["builtin_v1", "strong"], default="strong")
    parser.add_argument("--classic-factor-augment", dest="classic_factor_augment", action="store_true")
    parser.add_argument("--no-classic-factor-augment", dest="classic_factor_augment", action="store_false")
    parser.set_defaults(classic_factor_augment=True)
    parser.add_argument("--pretrain-loss-weights", type=str, default=DEFAULT_LOSS_WEIGHTS)
    parser.add_argument("--no-pretrain-aux-loss", action="store_true")
    parser.add_argument("--no-head-pretrain", action="store_true")
    parser.add_argument("--save-pretrain-ckpt", action="store_true")
    parser.add_argument("--pretrain-ckpt-path", type=str, default="")
    parser.add_argument("--load-pretrain-ckpt", type=str, default="")
    parser.add_argument("--motif-max-edits", type=int, default=4)
    parser.add_argument("--motif-prior-eta", type=float, default=0.02)
    parser.add_argument("--behavior-novelty-beta", type=float, default=0.05)
    parser.add_argument("--behavior-archive-size", type=int, default=128)
    parser.add_argument("--typed-robust-years", type=str, default="2014,2015,2016,2017,2018")
    parser.add_argument("--typed-robust-lambda", type=float, default=0.5)
    parser.add_argument("--typed-robust-bottom-k", type=int, default=0)
    parser.add_argument("--qd-cell-capacity", type=int, default=2)
    parser.add_argument("--qd-behavior-threshold", type=float, default=0.7)
    parser.add_argument("--qd-bonus", type=float, default=0.02)
    parser.add_argument("--typed-min-robust-score", type=float, default=-1.0)
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
        classic_factor_bank=args.classic_factor_bank,
        classic_factor_augment=args.classic_factor_augment,
        pretrain_loss_weights=args.pretrain_loss_weights,
        no_pretrain_aux_loss=args.no_pretrain_aux_loss,
        no_head_pretrain=args.no_head_pretrain,
        save_pretrain_ckpt=args.save_pretrain_ckpt,
        pretrain_ckpt_path=args.pretrain_ckpt_path,
        load_pretrain_ckpt=args.load_pretrain_ckpt,
        motif_max_edits=args.motif_max_edits,
        motif_prior_eta=args.motif_prior_eta,
        behavior_novelty_beta=args.behavior_novelty_beta,
        behavior_archive_size=args.behavior_archive_size,
        typed_robust_years=args.typed_robust_years,
        typed_robust_lambda=args.typed_robust_lambda,
        typed_robust_bottom_k=args.typed_robust_bottom_k,
        qd_cell_capacity=args.qd_cell_capacity,
        qd_behavior_threshold=args.qd_behavior_threshold,
        qd_bonus=args.qd_bonus,
        typed_min_robust_score=args.typed_min_robust_score,
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
