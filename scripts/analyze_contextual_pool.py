from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphagen.data.expression import Feature, FeatureType, Ref
from alphagen_context import (
    ASTGraphBuilder,
    ASTGraphEncoder,
    ContextAlphaEncoder,
    ContextEvaluator,
    DeepSetsCombiner,
    StructureClusterBank,
)
from alphagen_embedding.dataset import parse_expression_text
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.compat import patch_all
from alphagen_qlib.stock_data import StockData


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a contextual pool checkpoint.")
    parser.add_argument("--pretrain_bundle", type=str, required=True)
    parser.add_argument("--pool_json", type=str, required=True)
    parser.add_argument("--provider_uri", type=str, required=True)
    parser.add_argument("--market", type=str, default="csi300")
    parser.add_argument("--start_time", type=str, default="2021-01-01")
    parser.add_argument("--end_time", type=str, default="2022-12-31")
    parser.add_argument("--output_dir", type=str, default="")
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
    ast_encoder, alpha_encoder, combiner, bundle = build_context_modules(args.pretrain_bundle, device)
    evaluator = ContextEvaluator(
        ast_encoder=ast_encoder,
        alpha_encoder=alpha_encoder,
        combiner=combiner,
        graph_builder=ASTGraphBuilder(),
        lookback=int(bundle.get("config", {}).get("lookback", 60)),
        cluster_bank=StructureClusterBank(),
    )

    with open(args.pool_json, "r", encoding="utf-8") as f:
        pool_data = json.load(f)
    exprs = [parse_expression_text(text) for text in pool_data.get("exprs", [])]

    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1
    stock_data = StockData(instrument=args.market, start_time=args.start_time, end_time=args.end_time, device=device)
    calculator = QLibStockDataCalculator(stock_data, target)
    alpha_panels = torch.stack([calculator._calc_alpha(expr).detach().cpu() for expr in exprs], dim=0)
    target_panel = calculator.target_value.detach().cpu()
    eval_result = evaluator.evaluate_set(exprs, alpha_panels, target_panel)

    output_dir = Path(args.output_dir or (Path(args.pool_json).resolve().parent / "context_analysis"))
    output_dir.mkdir(parents=True, exist_ok=True)
    struct_embeddings = evaluator.encode_struct(exprs).detach().cpu()
    cluster_rows = []
    weights = eval_result["weights"].numpy().tolist()
    rows = []
    for idx, expr in enumerate(exprs):
        cluster_info = evaluator.cluster_bank.describe(struct_embeddings[idx])
        ri_reg = evaluator.compute_ri_reg(expr, alpha_panels[idx])
        rows.append(
            {
                "alpha_id": idx,
                "expr_text": str(expr),
                "weight": weights[idx],
                "ri_reg": float(ri_reg),
                "cluster_id": cluster_info.cluster_id,
                "cluster_value": cluster_info.cluster_value,
                "cluster_coverage": cluster_info.cluster_coverage,
                "cluster_under_explore": cluster_info.under_explore,
                **{f"emb_{j}": float(struct_embeddings[idx, j].item()) for j in range(struct_embeddings.shape[1])},
            }
        )
        updated = evaluator.cluster_bank.update(struct_embeddings[idx], float(weights[idx]))
        cluster_rows.append(
            {
                "alpha_id": idx,
                "expr_text": str(expr),
                "cluster_id": updated.cluster_id,
                "cluster_value": updated.cluster_value,
                "cluster_coverage": updated.cluster_coverage,
                "cluster_under_explore": updated.under_explore,
                "cluster_is_new": updated.is_new,
                "cluster_distance": updated.distance,
            }
        )
    alpha_csv = output_dir / "alpha_context_summary.csv"
    pd.DataFrame(rows).to_csv(alpha_csv, index=False)
    cluster_csv = output_dir / "cluster_summary.csv"
    pd.DataFrame(cluster_rows).to_csv(cluster_csv, index=False)
    pairwise_dist = torch.cdist(struct_embeddings, struct_embeddings, p=2).numpy()
    pairwise_csv = output_dir / "pairwise_struct_distance.csv"
    pd.DataFrame(pairwise_dist).to_csv(pairwise_csv, index=False)
    weight_png = output_dir / "weight_bar.png"
    heatmap_png = output_dir / "struct_distance_heatmap.png"
    scatter_png = output_dir / "struct_embedding_scatter.png"
    reward_png = output_dir / "reward_decomposition.png"

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(weights)), weights)
    plt.xlabel("alpha_id")
    plt.ylabel("weight")
    plt.title("Context Combiner Weights")
    plt.tight_layout()
    plt.savefig(weight_png, dpi=160)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.imshow(pairwise_dist, cmap="viridis", aspect="auto")
    plt.colorbar(label="L2 distance")
    plt.xlabel("alpha_id")
    plt.ylabel("alpha_id")
    plt.title("Pairwise Structure Distance")
    plt.tight_layout()
    plt.savefig(heatmap_png, dpi=160)
    plt.close()

    if struct_embeddings.shape[1] >= 2:
        scatter_x = struct_embeddings[:, 0].numpy()
        scatter_y = struct_embeddings[:, 1].numpy()
    else:
        scatter_x = list(range(struct_embeddings.shape[0]))
        scatter_y = [0.0] * struct_embeddings.shape[0]
    plt.figure(figsize=(7, 5))
    plt.scatter(scatter_x, scatter_y, c=weights, cmap="coolwarm", s=80)
    for idx, (x, y) in enumerate(zip(scatter_x, scatter_y)):
        plt.text(x, y, str(idx), fontsize=8)
    plt.xlabel("struct_dim_0")
    plt.ylabel("struct_dim_1")
    plt.title("Structure Embedding Projection")
    plt.tight_layout()
    plt.savefig(scatter_png, dpi=160)
    plt.close()

    reward_info = pool_data.get("last_reward_info", {}) or {}
    reward_keys = ["re", "ri_func", "ri_struct", "ri_reg", "reward_total"]
    reward_values = [float(reward_info.get(key, 0.0)) for key in reward_keys]
    plt.figure(figsize=(8, 4))
    plt.bar(reward_keys, reward_values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
    plt.ylabel("value")
    plt.title("Reward Decomposition")
    plt.tight_layout()
    plt.savefig(reward_png, dpi=160)
    plt.close()

    summary = {
        "metric": float(eval_result["metric"]),
        "rankic": float(eval_result["rankic"]),
        "sharpe": float(eval_result["sharpe"]),
        "turnover": float(eval_result["turnover"]),
        "weight_sparsity": float(eval_result["weight_sparsity"]),
        "alpha_csv": str(alpha_csv),
        "cluster_csv": str(cluster_csv),
        "pairwise_distance_csv": str(pairwise_csv),
        "weight_plot": str(weight_png),
        "distance_heatmap": str(heatmap_png),
        "embedding_scatter": str(scatter_png),
        "reward_decomposition_plot": str(reward_png),
        "last_reward_info": pool_data.get("last_reward_info", {}),
        "last_cluster_info": pool_data.get("last_cluster_info", {}),
        "pool_size": len(exprs),
    }
    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
