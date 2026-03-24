from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphagen.data.expression import Feature, FeatureType, Ref
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
    parser = argparse.ArgumentParser(description="Benchmark ContextAlphaPool evaluation latency.")
    parser.add_argument("--pretrain_bundle", type=str, required=True)
    parser.add_argument("--provider_uri", type=str, required=True)
    parser.add_argument("--checkpoint_root", type=str, default="kaggle/working/checkpoints")
    parser.add_argument("--runs_root", type=str, default="kaggle/working/runs")
    parser.add_argument("--alpha_source_file", type=str, default="")
    parser.add_argument("--market", type=str, default="csi300")
    parser.add_argument("--start_time", type=str, default="2021-01-01")
    parser.add_argument("--end_time", type=str, default="2021-12-31")
    parser.add_argument("--init_pool_size", type=int, default=5)
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument("--candidate_count", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


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
    return ast_encoder.eval(), alpha_encoder.eval(), combiner.eval(), payload


def main() -> None:
    args = parse_args()
    patch_all(args.provider_uri)
    import qlib
    from qlib.constant import REG_CN

    qlib.init(provider_uri=args.provider_uri, region=REG_CN)
    device = torch.device(args.device)
    ast_encoder, alpha_encoder, combiner, payload = build_context_modules(args.pretrain_bundle, device)
    evaluator = ContextEvaluator(
        ast_encoder=ast_encoder,
        alpha_encoder=alpha_encoder,
        combiner=combiner,
        graph_builder=ASTGraphBuilder(),
        lookback=int(payload.get("config", {}).get("lookback", 60)),
        cluster_bank=StructureClusterBank(),
    )

    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1
    stock_data = StockData(
        instrument=args.market,
        start_time=args.start_time,
        end_time=args.end_time,
        device=device,
    )
    calculator = QLibStockDataCalculator(stock_data, target)
    pool = ContextAlphaPool(capacity=args.capacity, calculator=calculator, evaluator=evaluator, device=device)

    if args.alpha_source_file:
        candidates = load_alpha_source_file(args.alpha_source_file)
        if len(candidates) < args.init_pool_size + args.candidate_count:
            raise ValueError(
                f"alpha_source_file only contains {len(candidates)} alphas, need at least {args.init_pool_size + args.candidate_count}"
            )
    else:
        candidates = collect_alpha_candidates(args.checkpoint_root, args.runs_root, min_alpha=args.init_pool_size + args.candidate_count)
    init_exprs = [parse_expression_text(row["expr_text"]) for row in candidates[: args.init_pool_size]]
    test_exprs = [parse_expression_text(row["expr_text"]) for row in candidates[args.init_pool_size : args.init_pool_size + args.candidate_count]]

    t0 = time.perf_counter()
    pool.warm_start(init_exprs)
    warm_start_sec = time.perf_counter() - t0

    timings = []
    for expr in test_exprs:
        t1 = time.perf_counter()
        reward, info = pool.try_new_expr(expr)
        elapsed = time.perf_counter() - t1
        timings.append(
            {
                "expr": str(expr),
                "elapsed_sec": elapsed,
                "reward": float(reward),
                "re": float(info.get("re", 0.0)),
                "ri_func": float(info.get("ri_func", 0.0)),
                "ri_struct": float(info.get("ri_struct", 0.0)),
                "ri_reg": float(info.get("ri_reg", 0.0)),
            }
        )

    summary = {
        "warm_start_sec": warm_start_sec,
        "candidate_count": len(timings),
        "mean_candidate_sec": sum(item["elapsed_sec"] for item in timings) / max(1, len(timings)),
        "timings": timings,
        "pool_size_after": pool.size,
        "best_metric": pool.best_metric,
        "best_rankic": pool.best_rankic,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
