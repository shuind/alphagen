from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphagen.data.expression import Feature, FeatureType, Ref
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet, TransformerSharedNet
from alphagen.utils.random import reseed_everything
from alphagen_context import (
    ASTGraphBuilder,
    ASTGraphEncoder,
    ContextAlphaEncoder,
    ContextAlphaPool,
    ContextEvaluator,
    DeepSetsCombiner,
    StructureClusterBank,
)
from alphagen_embedding.dataset import collect_alpha_candidates, load_alpha_source_file, parse_expression_text
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.compat import patch_all
from alphagen_qlib.stock_data import StockData


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Context-Aware AlphaGen PPO line.")
    parser.add_argument("--pretrain_bundle", type=str, required=True)
    parser.add_argument("--provider_uri", type=str, required=True)
    parser.add_argument("--market", type=str, default="csi300")
    parser.add_argument("--pool_capacity", type=int, default=30)
    parser.add_argument("--init_pool_size", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--n_steps", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backbone", type=str, default="lstm", choices=["lstm", "transformer"])
    parser.add_argument("--checkpoint_root", type=str, default="kaggle/working/checkpoints")
    parser.add_argument("--runs_root", type=str, default="kaggle/working/runs")
    parser.add_argument("--alpha_source_file", type=str, default="")
    parser.add_argument("--reward_lambda", type=float, default=0.3)
    parser.add_argument("--reward_schedule_decay", type=float, default=0.0)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--xi", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_root", type=str, default="context_runs")
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_context_modules(bundle_path: str, device: torch.device):
    payload = torch.load(bundle_path, map_location=device)
    config = payload.get("config", {})
    ast_encoder = ASTGraphEncoder(
        hidden_dim=int(config.get("ast_hidden_dim", 64)),
        output_dim=int(config.get("ast_output_dim", 32)),
    ).to(device)
    alpha_encoder = ContextAlphaEncoder(
        behavior_hidden_size=int(config.get("behavior_hidden_size", 32)),
        stat_hidden_dim=int(config.get("stat_hidden_dim", 16)),
        output_dim=int(config.get("alpha_output_dim", 32)),
    ).to(device)
    combiner = DeepSetsCombiner(
        input_dim=int(config.get("ast_output_dim", 32)) + int(config.get("alpha_output_dim", 32)),
        hidden_dim=int(config.get("combiner_hidden_dim", 64)),
        activation=str(config.get("combiner_activation", "sparsemax")),
    ).to(device)
    ast_encoder.load_state_dict(payload["ast_encoder_state_dict"])
    alpha_encoder.load_state_dict(payload["alpha_encoder_state_dict"])
    combiner.load_state_dict(payload["combiner_state_dict"])
    ast_encoder.eval()
    alpha_encoder.eval()
    combiner.eval()
    return ast_encoder, alpha_encoder, combiner, payload


def log_metrics_csv(run_dir: str, step: int, metrics: dict) -> None:
    path = Path(run_dir) / "metrics_step.csv"
    write_header = not path.is_file()
    fieldnames = [
        "step",
        "pool_size",
        "best_metric",
        "best_rankic",
        "test_metric",
        "test_rankic",
        "RE",
        "RI_func",
        "RI_struct",
        "RI_reg",
        "marginal_rankic_gain",
        "combiner_weight_sparsity",
        "cluster_id",
        "cluster_value",
        "cluster_coverage",
    ]
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        row = {key: metrics.get(key, math.nan) for key in fieldnames}
        row["step"] = step
        writer.writerow(row)


class ContextCallback(BaseCallback):
    def __init__(self, save_path: str, run_dir: str, test_calculator: QLibStockDataCalculator, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.run_dir = run_dir
        self.test_calculator = test_calculator

    @property
    def pool(self) -> ContextAlphaPool:
        return self.training_env.envs[0].unwrapped.pool  # type: ignore

    def _init_callback(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.run_dir, exist_ok=True)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        metric_test, rankic_test = self.pool.test_ensemble(self.test_calculator)
        info = dict(self.pool.last_reward_info)
        info.update(self.pool.last_cluster_info)
        info.update(
            {
                "pool_size": self.pool.size,
                "best_metric": self.pool.best_metric,
                "best_rankic": self.pool.best_rankic,
                "test_metric": metric_test,
                "test_rankic": rankic_test,
                "RE": info.get("re", math.nan),
                "RI_func": info.get("ri_func", math.nan),
                "RI_struct": info.get("ri_struct", math.nan),
                "RI_reg": info.get("ri_reg", math.nan),
            }
        )
        log_metrics_csv(self.run_dir, self.num_timesteps, info)
        self.logger.record("context/best_metric", self.pool.best_metric)
        self.logger.record("context/best_rankic", self.pool.best_rankic)
        self.logger.record("test/metric", metric_test)
        self.logger.record("test/rank_ic", rankic_test)
        self.model.save(os.path.join(self.save_path, f"{self.num_timesteps}_steps"))  # type: ignore[attr-defined]
        with open(os.path.join(self.save_path, f"{self.num_timesteps}_steps_pool.json"), "w", encoding="utf-8") as f:
            json.dump(self.pool.to_dict(), f)


def main() -> None:
    args = parse_args()
    reseed_everything(args.seed)
    patch_all(args.provider_uri)
    import qlib
    from qlib.constant import REG_CN

    qlib.init(provider_uri=args.provider_uri, region=REG_CN)
    device = resolve_device(args.device)

    ast_encoder, alpha_encoder, combiner, bundle = build_context_modules(args.pretrain_bundle, device)
    evaluator = ContextEvaluator(
        ast_encoder=ast_encoder,
        alpha_encoder=alpha_encoder,
        combiner=combiner,
        graph_builder=ASTGraphBuilder(),
        lookback=int(bundle.get("config", {}).get("lookback", 60)),
        eta=args.eta,
        xi=args.xi,
        cluster_bank=StructureClusterBank(),
        reward_lambda=args.reward_lambda,
        reward_schedule_decay=args.reward_schedule_decay,
    )

    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1
    data_train = StockData(instrument=args.market, start_time="2010-01-01", end_time="2019-12-31", device=device)
    data_test = StockData(instrument=args.market, start_time="2021-01-01", end_time="2022-12-31", device=device)
    calculator_train = QLibStockDataCalculator(data_train, target)
    calculator_test = QLibStockDataCalculator(data_test, target)

    pool = ContextAlphaPool(capacity=args.pool_capacity, calculator=calculator_train, evaluator=evaluator, device=device)
    if args.alpha_source_file:
        candidates = load_alpha_source_file(args.alpha_source_file)
        if len(candidates) < args.init_pool_size:
            raise ValueError(f"alpha_source_file only contains {len(candidates)} alphas, need at least {args.init_pool_size}")
    else:
        candidates = collect_alpha_candidates(args.checkpoint_root, args.runs_root, min_alpha=args.init_pool_size, max_alpha=max(args.init_pool_size, 50))
    init_exprs = [parse_expression_text(row["expr_text"]) for row in candidates[: args.init_pool_size]]
    pool.warm_start(init_exprs)
    env = AlphaEnv(pool=pool, device=device, print_expr=True, reward_per_step=0.0)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_id = f"context_seed{args.seed}_{args.backbone}_{timestamp}"
    run_dir = Path(args.output_root) / run_id
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "market": args.market,
                "seed": args.seed,
                "steps": args.steps,
                "generator_backbone": args.backbone,
                "ast_encoder": "ASTGraphEncoder",
                "alpha_encoder": "ContextAlphaEncoder",
                "combiner": "DeepSetsCombiner",
                "reward_components": ["RE", "RI_func", "RI_struct", "RI_reg"],
                "reward_schedule": {
                    "reward_lambda": args.reward_lambda,
                    "reward_schedule_decay": args.reward_schedule_decay,
                },
                "pretrain_bundle": args.pretrain_bundle,
                "init_pool_size": args.init_pool_size,
                "pool_capacity": args.pool_capacity,
                "alpha_source_file": args.alpha_source_file,
            },
            f,
            indent=2,
        )

    if args.backbone == "transformer":
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
    policy_kwargs = {
        "features_extractor_class": features_extractor_class,
        "features_extractor_kwargs": features_extractor_kwargs,
    }
    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        gamma=1.0,
        n_steps=args.n_steps,
        ent_coef=args.ent_coef,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        clip_range=args.clip_range,
        verbose=1,
        device=device,
        tensorboard_log=str(run_dir / "tb"),
    )
    callback = ContextCallback(str(ckpt_dir), str(run_dir), calculator_test)
    model.learn(total_timesteps=args.steps, callback=callback, tb_log_name="contextual")


if __name__ == "__main__":
    main()
