from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphagen.utils.correlation import batch_pearsonr
from alphagen_context.alpha_encoder import ContextAlphaEncoder
from alphagen_context.ast_encoder import ASTGraphBuilder, ASTGraphEncoder
from alphagen_context.combiner import DeepSetsCombiner
from alphagen_embedding.dataset import (
    build_embedding_dataset,
    load_alpha_source_file,
    parse_expression_text,
    save_embedding_dataset,
)


class EmbeddingTensorDataset(Dataset):
    def __init__(self, dataset: dict, split: str):
        indices = dataset["split_indices"][split]
        self.behavior = dataset["behavior"][indices]
        self.stats = dataset["stats"][indices]
        self.alpha_values = dataset["alpha_values"][indices]
        self.targets = dataset["targets"][indices]

    def __len__(self) -> int:
        return int(self.behavior.shape[0])

    def __getitem__(self, idx: int):
        return (
            self.behavior[idx],
            self.stats[idx],
            self.alpha_values[idx],
            self.targets[idx],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain AST-GNN + alpha encoder + DeepSets combiner.")
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--checkpoint_root", type=str, default="kaggle/working/checkpoints")
    parser.add_argument("--runs_root", type=str, default="kaggle/working/runs")
    parser.add_argument("--provider_uri", type=str, default="")
    parser.add_argument("--alpha_source_file", type=str, default="")
    parser.add_argument("--output_root", type=str, default="context_outputs")
    parser.add_argument("--market", type=str, default="csi300")
    parser.add_argument("--start_time", type=str, default="2021-01-01")
    parser.add_argument("--end_time", type=str, default="2022-12-31")
    parser.add_argument("--num_alpha", type=int, default=20)
    parser.add_argument("--stock_limit", type=int, default=100)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ast_hidden_dim", type=int, default=64)
    parser.add_argument("--ast_output_dim", type=int, default=32)
    parser.add_argument("--behavior_hidden_size", type=int, default=32)
    parser.add_argument("--stat_hidden_dim", type=int, default=16)
    parser.add_argument("--alpha_output_dim", type=int, default=32)
    parser.add_argument("--combiner_hidden_dim", type=int, default=64)
    parser.add_argument("--combiner_activation", type=str, default="sparsemax", choices=["sparsemax", "softmax"])
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_or_load_dataset(args: argparse.Namespace, output_dir: Path) -> dict:
    if args.dataset_path:
        return torch.load(args.dataset_path, map_location="cpu")
    alpha_candidates = load_alpha_source_file(args.alpha_source_file) if args.alpha_source_file else None
    dataset = build_embedding_dataset(
        checkpoint_root=args.checkpoint_root,
        runs_root=args.runs_root,
        provider_uri=args.provider_uri,
        market=args.market,
        start_time=args.start_time,
        end_time=args.end_time,
        num_alpha=args.num_alpha,
        stock_limit=args.stock_limit,
        lookback=args.lookback,
        horizon=args.horizon,
        device="cpu",
        alpha_candidates=alpha_candidates,
    )
    save_embedding_dataset(dataset, output_dir)
    return dataset


@torch.no_grad()
def evaluate_split(ast_encoder, alpha_encoder, combiner, graph_batch, loader, device):
    ast_encoder.eval()
    alpha_encoder.eval()
    combiner.eval()
    struct_embed = ast_encoder(graph_batch.to(device))
    losses = []
    ics = []
    for behavior, stats, alpha_values, targets in loader:
        behavior = behavior.to(device)
        stats = stats.to(device)
        alpha_values = alpha_values.to(device)
        targets = targets.to(device)
        z_alpha = alpha_encoder(behavior, stats)
        z_struct = struct_embed.unsqueeze(0).expand(behavior.shape[0], -1, -1)
        outputs = combiner(torch.cat([z_alpha, z_struct], dim=-1), alpha_values)
        ic = batch_pearsonr(outputs["prediction"], targets).mean()
        losses.append(float((-ic).item()))
        ics.append(float(ic.item()))
    return {
        "loss": float(sum(losses) / max(1, len(losses))),
        "ic": float(sum(ics) / max(1, len(ics))),
    }


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"context_pretrain_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_or_load_dataset(args, output_dir)
    device = resolve_device(args.device)

    graph_builder = ASTGraphBuilder()
    graph_batch = graph_builder.build_batch([parse_expression_text(text) for text in dataset["expr_texts"]])
    ast_encoder = ASTGraphEncoder(hidden_dim=args.ast_hidden_dim, output_dim=args.ast_output_dim).to(device)
    alpha_encoder = ContextAlphaEncoder(
        behavior_hidden_size=args.behavior_hidden_size,
        stat_hidden_dim=args.stat_hidden_dim,
        output_dim=args.alpha_output_dim,
    ).to(device)
    combiner = DeepSetsCombiner(
        input_dim=args.ast_output_dim + args.alpha_output_dim,
        hidden_dim=args.combiner_hidden_dim,
        activation=args.combiner_activation,
    ).to(device)

    train_loader = DataLoader(EmbeddingTensorDataset(dataset, "train"), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(EmbeddingTensorDataset(dataset, "valid"), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(EmbeddingTensorDataset(dataset, "test"), batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(
        list(ast_encoder.parameters()) + list(alpha_encoder.parameters()) + list(combiner.parameters()),
        lr=args.lr,
    )
    best_valid_ic = float("-inf")
    bundle_path = output_dir / "context_bundle.pt"
    log_path = output_dir / "pretrain_log.csv"

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_ic", "valid_loss", "valid_ic"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            ast_encoder.train()
            alpha_encoder.train()
            combiner.train()
            train_losses = []
            train_ics = []
            for behavior, stats, alpha_values, targets in train_loader:
                struct_embed = ast_encoder(graph_batch.to(device))
                behavior = behavior.to(device)
                stats = stats.to(device)
                alpha_values = alpha_values.to(device)
                targets = targets.to(device)
                z_alpha = alpha_encoder(behavior, stats)
                z_struct = struct_embed.unsqueeze(0).expand(behavior.shape[0], -1, -1)
                outputs = combiner(torch.cat([z_alpha, z_struct], dim=-1), alpha_values)
                ic = batch_pearsonr(outputs["prediction"], targets).mean()
                loss = -ic
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.item()))
                train_ics.append(float(ic.item()))
            valid_metrics = evaluate_split(ast_encoder, alpha_encoder, combiner, graph_batch, valid_loader, device)
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": sum(train_losses) / max(1, len(train_losses)),
                    "train_ic": sum(train_ics) / max(1, len(train_ics)),
                    "valid_loss": valid_metrics["loss"],
                    "valid_ic": valid_metrics["ic"],
                }
            )
            if valid_metrics["ic"] > best_valid_ic:
                best_valid_ic = valid_metrics["ic"]
                torch.save(
                    {
                        "ast_encoder_state_dict": ast_encoder.state_dict(),
                        "alpha_encoder_state_dict": alpha_encoder.state_dict(),
                        "combiner_state_dict": combiner.state_dict(),
                        "expr_texts": dataset["expr_texts"],
                        "alpha_summary": dataset["alpha_summary"],
                        "config": vars(args),
                        "best_valid_ic": best_valid_ic,
                    },
                    bundle_path,
                )

    bundle = torch.load(bundle_path, map_location=device)
    ast_encoder.load_state_dict(bundle["ast_encoder_state_dict"])
    alpha_encoder.load_state_dict(bundle["alpha_encoder_state_dict"])
    combiner.load_state_dict(bundle["combiner_state_dict"])
    valid_metrics = evaluate_split(ast_encoder, alpha_encoder, combiner, graph_batch, valid_loader, device)
    test_metrics = evaluate_split(ast_encoder, alpha_encoder, combiner, graph_batch, test_loader, device)
    summary = {
        "bundle_path": str(bundle_path),
        "dataset_path": str(output_dir / "embedding_dataset.pt"),
        "log_path": str(log_path),
        "best_valid_ic": float(best_valid_ic),
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "config": vars(args),
    }
    with (output_dir / "pretrain_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
