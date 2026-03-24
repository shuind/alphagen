from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import Tensor

from alphagen.utils.correlation import batch_pearsonr


def export_mean_embeddings(
    embeddings: Tensor,
    alpha_summary: List[Dict[str, object]],
    output_dir: str | Path,
    split_name: str,
) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    emb_np = embeddings.detach().cpu().numpy()
    for idx, summary in enumerate(alpha_summary):
        row = dict(summary)
        for dim_idx, value in enumerate(emb_np[idx]):
            row[f"emb_{dim_idx}"] = float(value)
        row["split"] = split_name
        rows.append(row)
    df = pd.DataFrame(rows)
    path = output_dir / f"embedding_export_{split_name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def _plot_embedding_projection(
    coords: np.ndarray,
    labels: List[str],
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 7))
    plt.scatter(coords[:, 0], coords[:, 1], s=50, alpha=0.85)
    for idx, label in enumerate(labels):
        plt.annotate(str(idx), (coords[idx, 0], coords[idx, 1]), fontsize=8)
    plt.title(title)
    plt.xlabel("dim1")
    plt.ylabel("dim2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_embedding_visualizations(
    embeddings: Tensor,
    alpha_summary: List[Dict[str, object]],
    output_dir: str | Path,
    split_name: str,
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    emb_np = embeddings.detach().cpu().numpy()
    labels = [str(item.get("alpha_id", idx)) for idx, item in enumerate(alpha_summary)]

    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(emb_np)
    pca_path = output_dir / f"pca_{split_name}.png"
    _plot_embedding_projection(pca_coords, labels, f"PCA - {split_name}", pca_path)

    perplexity = max(2, min(10, emb_np.shape[0] - 1))
    tsne = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto", perplexity=perplexity)
    tsne_coords = tsne.fit_transform(emb_np)
    tsne_path = output_dir / f"tsne_{split_name}.png"
    _plot_embedding_projection(tsne_coords, labels, f"t-SNE - {split_name}", tsne_path)

    pd.DataFrame(pca_coords, columns=["x", "y"]).assign(alpha_id=labels).to_csv(
        output_dir / f"pca_{split_name}.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(tsne_coords, columns=["x", "y"]).assign(alpha_id=labels).to_csv(
        output_dir / f"tsne_{split_name}.csv", index=False, encoding="utf-8-sig"
    )
    return {"pca": str(pca_path), "tsne": str(tsne_path)}


def nearest_neighbor_analysis(
    embeddings: Tensor,
    alpha_summary: List[Dict[str, object]],
    output_dir: str | Path,
    split_name: str,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    emb_np = embeddings.detach().cpu().numpy()
    pairwise = np.linalg.norm(emb_np[:, None, :] - emb_np[None, :, :], axis=2)
    np.fill_diagonal(pairwise, np.inf)
    neighbors = pairwise.argmin(axis=1)

    rows = []
    ic_diffs = []
    rankic_diffs = []
    return_diffs = []
    for idx, neighbor_idx in enumerate(neighbors):
        src = alpha_summary[idx]
        dst = alpha_summary[int(neighbor_idx)]
        ic_diff = abs(float(src["mean_ic"]) - float(dst["mean_ic"]))
        rankic_diff = abs(float(src["mean_rankic"]) - float(dst["mean_rankic"]))
        return_diff = abs(float(src["mean_return"]) - float(dst["mean_return"]))
        ic_diffs.append(ic_diff)
        rankic_diffs.append(rankic_diff)
        return_diffs.append(return_diff)
        rows.append(
            {
                "alpha_id": src["alpha_id"],
                "neighbor_alpha_id": dst["alpha_id"],
                "distance": float(pairwise[idx, neighbor_idx]),
                "mean_ic_diff": ic_diff,
                "mean_rankic_diff": rankic_diff,
                "mean_return_diff": return_diff,
            }
        )

    nn_path = output_dir / f"nearest_neighbors_{split_name}.csv"
    pd.DataFrame(rows).to_csv(nn_path, index=False, encoding="utf-8-sig")
    summary = {
        "split": split_name,
        "mean_neighbor_distance": float(np.mean([row["distance"] for row in rows])),
        "mean_ic_abs_diff": float(np.mean(ic_diffs)),
        "mean_rankic_abs_diff": float(np.mean(rankic_diffs)),
        "mean_return_abs_diff": float(np.mean(return_diffs)),
        "neighbor_csv": str(nn_path),
    }
    summary_path = output_dir / f"nearest_neighbors_{split_name}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def _performance_score(alpha_summary: List[Dict[str, object]]) -> np.ndarray:
    scores = []
    for item in alpha_summary:
        mean_ic = float(item.get("mean_ic", 0.0))
        mean_rankic = float(item.get("mean_rankic", 0.0))
        mean_return = float(item.get("mean_return", 0.0))
        scores.append(mean_ic + 0.5 * mean_rankic + 0.5 * mean_return)
    return np.asarray(scores, dtype=np.float32)


def embedding_deduplicate(
    embeddings: Tensor,
    alpha_summary: List[Dict[str, object]],
    output_dir: str | Path,
    split_name: str,
    distance_threshold: Optional[float] = None,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    emb_np = embeddings.detach().cpu().numpy()
    pairwise = np.linalg.norm(emb_np[:, None, :] - emb_np[None, :, :], axis=2)
    np.fill_diagonal(pairwise, np.inf)
    scores = _performance_score(alpha_summary)
    finite_distances = pairwise[np.isfinite(pairwise)]
    if distance_threshold is None:
        distance_threshold = float(np.quantile(finite_distances, 0.25)) if finite_distances.size > 0 else 0.0

    order = np.argsort(-scores)
    selected: List[int] = []
    removed_rows: List[Dict[str, object]] = []
    selected_mask = np.zeros(len(alpha_summary), dtype=bool)
    for idx in order:
        if selected_mask[idx]:
            continue
        selected.append(int(idx))
        selected_mask[idx] = True
        close_indices = np.where(pairwise[idx] <= distance_threshold)[0]
        for neighbor_idx in close_indices:
            if selected_mask[int(neighbor_idx)]:
                continue
            selected_mask[int(neighbor_idx)] = True
            removed_rows.append(
                {
                    "kept_alpha_id": int(alpha_summary[int(idx)]["alpha_id"]),
                    "removed_alpha_id": int(alpha_summary[int(neighbor_idx)]["alpha_id"]),
                    "distance": float(pairwise[idx, neighbor_idx]),
                    "kept_score": float(scores[idx]),
                    "removed_score": float(scores[int(neighbor_idx)]),
                }
            )

    kept_rows = []
    for idx in selected:
        row = dict(alpha_summary[idx])
        row["selection_score"] = float(scores[idx])
        kept_rows.append(row)

    kept_path = output_dir / f"dedup_kept_{split_name}.csv"
    removed_path = output_dir / f"dedup_removed_{split_name}.csv"
    pd.DataFrame(kept_rows).to_csv(kept_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(removed_rows).to_csv(removed_path, index=False, encoding="utf-8-sig")

    summary = {
        "split": split_name,
        "distance_threshold": float(distance_threshold),
        "kept_count": len(kept_rows),
        "removed_count": len(removed_rows),
        "kept_csv": str(kept_path),
        "removed_csv": str(removed_path),
    }
    summary_path = output_dir / f"dedup_summary_{split_name}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def embedding_diverse_selection(
    embeddings: Tensor,
    alpha_summary: List[Dict[str, object]],
    output_dir: str | Path,
    split_name: str,
    select_k: int = 10,
    lambda_diversity: float = 0.35,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    emb_np = embeddings.detach().cpu().numpy()
    pairwise = np.linalg.norm(emb_np[:, None, :] - emb_np[None, :, :], axis=2)
    np.fill_diagonal(pairwise, np.inf)
    scores = _performance_score(alpha_summary)

    remaining = set(range(len(alpha_summary)))
    selected: List[int] = []
    first_idx = int(np.argmax(scores))
    selected.append(first_idx)
    remaining.remove(first_idx)

    while remaining and len(selected) < min(select_k, len(alpha_summary)):
        best_idx = None
        best_objective = -float("inf")
        for idx in remaining:
            min_distance = float(np.min(pairwise[idx, selected])) if selected else 0.0
            objective = float(scores[idx] + lambda_diversity * min_distance)
            if objective > best_objective:
                best_objective = objective
                best_idx = idx
        assert best_idx is not None
        selected.append(int(best_idx))
        remaining.remove(int(best_idx))

    rows = []
    for rank, idx in enumerate(selected, start=1):
        row = dict(alpha_summary[idx])
        row["selection_rank"] = rank
        row["selection_score"] = float(scores[idx])
        row["min_distance_to_selected"] = (
            float(np.min(pairwise[idx, [s for s in selected if s != idx]]))
            if len(selected) > 1 and any(s != idx for s in selected)
            else None
        )
        rows.append(row)

    path = output_dir / f"diverse_selection_{split_name}.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    summary = {
        "split": split_name,
        "select_k": int(min(select_k, len(alpha_summary))),
        "lambda_diversity": float(lambda_diversity),
        "selected_alpha_ids": [int(alpha_summary[idx]["alpha_id"]) for idx in selected],
        "selection_csv": str(path),
    }
    summary_path = output_dir / f"diverse_selection_{split_name}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def _subset_ic(alpha_values: Tensor, targets: Tensor, alpha_indices: List[int], weight_mode: str, train_stats: Tensor) -> float:
    subset = alpha_values[:, alpha_indices, :]
    if weight_mode == "equal":
        pred = subset.mean(dim=1)
    elif weight_mode == "train_mean_ic":
        weights = train_stats[alpha_indices]
        weights = torch.softmax(weights, dim=0)
        pred = (subset * weights.view(1, -1, 1)).sum(dim=1)
    else:
        raise ValueError(f"Unknown weight mode: {weight_mode}")
    return float(batch_pearsonr(pred, targets).mean().item())


def compare_alpha_subsets(
    dataset: Dict[str, object],
    dedup_summary: Dict[str, object],
    diverse_summary: Dict[str, object],
    output_dir: str | Path,
) -> Dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    alpha_summary = dataset["alpha_summary"]
    full_indices = list(range(len(alpha_summary)))
    dedup_df = pd.read_csv(dedup_summary["kept_csv"])
    diverse_df = pd.read_csv(diverse_summary["selection_csv"])
    dedup_indices = [int(x) for x in dedup_df["alpha_id"].tolist()]
    diverse_indices = [int(x) for x in diverse_df["alpha_id"].tolist()]

    train_idx = torch.tensor(dataset["split_indices"]["train"], dtype=torch.long)
    train_stats = dataset["stats"][train_idx, :, 0].mean(dim=0)

    rows = []
    for split_name in ["train", "valid", "test"]:
        idx = torch.tensor(dataset["split_indices"][split_name], dtype=torch.long)
        alpha_values = dataset["alpha_values"][idx]
        targets = dataset["targets"][idx]
        for subset_name, alpha_indices in [
            ("full_pool", full_indices),
            ("dedup_kept", dedup_indices),
            ("diverse_selection", diverse_indices),
        ]:
            rows.append(
                {
                    "split": split_name,
                    "subset": subset_name,
                    "alpha_count": len(alpha_indices),
                    "equal_weight_ic": _subset_ic(alpha_values, targets, alpha_indices, "equal", train_stats),
                    "train_mean_ic_weighted_ic": _subset_ic(alpha_values, targets, alpha_indices, "train_mean_ic", train_stats),
                }
            )

    df = pd.DataFrame(rows)
    csv_path = output_dir / "subset_ic_comparison.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary = {"comparison_csv": str(csv_path), "rows": rows}
    json_path = output_dir / "subset_ic_comparison.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary
