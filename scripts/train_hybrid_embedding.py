import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alphagen.utils.correlation import batch_pearsonr
from alphagen_embedding.analysis import export_mean_embeddings
from alphagen_embedding.dataset import build_embedding_dataset, save_embedding_dataset
from alphagen_embedding.model import HybridAlphaEmbeddingModel


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def ic_loss(prediction: Tensor, target: Tensor) -> Tensor:
    return -batch_pearsonr(prediction, target).mean()


def assert_finite(name: str, tensor: Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")


def slice_split(dataset: Dict[str, object], split_name: str) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    indices = dataset["split_indices"][split_name]
    idx = torch.tensor(indices, dtype=torch.long)
    return (
        dataset["behavior"][idx],
        dataset["stats"][idx],
        dataset["alpha_values"][idx],
        dataset["targets"][idx],
    )


def build_dataloader(dataset: Dict[str, object], split_name: str, batch_size: int, shuffle: bool) -> DataLoader:
    split_tensors = slice_split(dataset, split_name)
    tensor_dataset = TensorDataset(*split_tensors)
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_model(model: HybridAlphaEmbeddingModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    losses = []
    ics = []
    with torch.no_grad():
        for behavior, stats, alpha_values, target in loader:
            behavior = behavior.to(device)
            stats = stats.to(device)
            alpha_values = alpha_values.to(device)
            target = target.to(device)
            output = model(behavior, stats, alpha_values)
            loss = ic_loss(output["prediction"], target)
            losses.append(float(loss.item()))
            ics.append(float(batch_pearsonr(output["prediction"], target).mean().item()))
    return {
        "loss": float(sum(losses) / max(1, len(losses))),
        "ic": float(sum(ics) / max(1, len(ics))),
    }


def baseline_metrics(dataset: Dict[str, object], split_name: str, train_static_weights: Tensor | None = None) -> Dict[str, float]:
    _, stats, alpha_values, targets = slice_split(dataset, split_name)
    equal_pred = alpha_values.mean(dim=1)
    equal_ic = float(batch_pearsonr(equal_pred, targets).mean().item())
    metrics = {"equal_weight_ic": equal_ic}
    if train_static_weights is not None:
        pred = (alpha_values * train_static_weights.view(1, -1, 1)).sum(dim=1)
        metrics["static_mean_ic_weighted_ic"] = float(batch_pearsonr(pred, targets).mean().item())
    return metrics


def save_training_log(rows: list[dict], output_dir: Path) -> str:
    path = output_dir / "train_log.csv"
    if not rows:
        return str(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="")
    parser.add_argument("--checkpoint_root", default="kaggle/working/checkpoints")
    parser.add_argument("--runs_root", default="kaggle/working/runs")
    parser.add_argument("--provider_uri", default=os.environ.get("QLIB_PROVIDER_URI", ""))
    parser.add_argument("--output_root", default="embedding_outputs")
    parser.add_argument("--market", default="csi300")
    parser.add_argument("--start_time", default="2021-01-01")
    parser.add_argument("--end_time", default="2022-12-31")
    parser.add_argument("--num_alpha", type=int, default=20)
    parser.add_argument("--stock_limit", type=int, default=100)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--behavior_hidden_size", type=int, default=32)
    parser.add_argument("--stat_hidden_dim", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    output_dir = Path(args.output_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    if args.dataset_path:
        dataset = torch.load(args.dataset_path, map_location="cpu")
        dataset_paths = {"dataset_path": args.dataset_path}
    else:
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
        )
        dataset_paths = save_embedding_dataset(dataset, output_dir)

    model = HybridAlphaEmbeddingModel(
        behavior_hidden_size=args.behavior_hidden_size,
        stat_hidden_dim=args.stat_hidden_dim,
        embedding_dim=args.embedding_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = build_dataloader(dataset, "train", args.batch_size, shuffle=True)
    valid_loader = build_dataloader(dataset, "valid", args.batch_size, shuffle=False)
    test_loader = build_dataloader(dataset, "test", args.batch_size, shuffle=False)
    split_sizes = {key: len(value) for key, value in dataset["split_indices"].items()}
    if split_sizes["train"] == 0:
        raise ValueError("Embedding dataset has no train samples. Adjust lookback/date range.")

    train_mean_ic = dataset["stats"][torch.tensor(dataset["split_indices"]["train"]), :, 0].mean(dim=0)
    static_weights = torch.softmax(train_mean_ic, dim=0)

    best_valid_ic = float("-inf")
    best_ckpt_path = output_dir / "best_model.pt"
    train_log_rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        batch_ics = []
        for behavior, stats, alpha_values, target in train_loader:
            behavior = behavior.to(device)
            stats = stats.to(device)
            alpha_values = alpha_values.to(device)
            target = target.to(device)
            assert_finite("behavior", behavior)
            assert_finite("stats", stats)
            assert_finite("alpha_values", alpha_values)
            assert_finite("target", target)

            output = model(behavior, stats, alpha_values)
            assert_finite("prediction", output["prediction"])
            loss = ic_loss(output["prediction"], target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(float(loss.item()))
            batch_ics.append(float(batch_pearsonr(output["prediction"], target).mean().item()))

        train_metrics = {
            "loss": float(sum(batch_losses) / max(1, len(batch_losses))),
            "ic": float(sum(batch_ics) / max(1, len(batch_ics))),
        }
        valid_metrics = evaluate_model(model, valid_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_ic": train_metrics["ic"],
            "valid_loss": valid_metrics["loss"],
            "valid_ic": valid_metrics["ic"],
        }
        train_log_rows.append(row)

        if valid_metrics["ic"] > best_valid_ic:
            best_valid_ic = valid_metrics["ic"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(args),
                    "best_valid_ic": best_valid_ic,
                    "expr_texts": dataset["expr_texts"],
                    "alpha_summary": dataset["alpha_summary"],
                },
                best_ckpt_path,
            )

    if not best_ckpt_path.is_file():
        fallback_ic = train_log_rows[-1]["valid_ic"] if train_log_rows else float("nan")
        best_valid_ic = fallback_ic
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": vars(args),
                "best_valid_ic": best_valid_ic,
                "expr_texts": dataset["expr_texts"],
                "alpha_summary": dataset["alpha_summary"],
                "note": "fallback_checkpoint_saved_without_valid_improvement",
            },
            best_ckpt_path,
        )

    train_log_path = save_training_log(train_log_rows, output_dir)

    best_payload = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_payload["model_state_dict"])

    valid_metrics = evaluate_model(model, valid_loader, device)
    test_metrics = evaluate_model(model, test_loader, device)
    train_baselines = baseline_metrics(dataset, "train", static_weights)
    valid_baselines = baseline_metrics(dataset, "valid", static_weights)
    test_baselines = baseline_metrics(dataset, "test", static_weights)

    model.eval()
    split_embedding_exports = {}
    with torch.no_grad():
        for split_name in ["train", "valid", "test"]:
            behavior, stats, alpha_values, _ = slice_split(dataset, split_name)
            output = model(
                behavior.to(device),
                stats.to(device),
                alpha_values.to(device),
            )
            mean_embeddings = output["embeddings"].mean(dim=0).cpu()
            export_path = export_mean_embeddings(mean_embeddings, dataset["alpha_summary"], output_dir, split_name)
            split_embedding_exports[split_name] = {"csv": export_path}
            torch.save(mean_embeddings, output_dir / f"mean_embeddings_{split_name}.pt")

    summary = {
        "dataset_path": dataset_paths["dataset_path"],
        "best_checkpoint": str(best_ckpt_path),
        "train_log_path": train_log_path,
        "best_valid_ic": best_valid_ic,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "train_baselines": train_baselines,
        "valid_baselines": valid_baselines,
        "test_baselines": test_baselines,
        "embedding_exports": split_embedding_exports,
        "config": vars(args),
        "split_sizes": split_sizes,
    }
    summary_path = output_dir / "train_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
