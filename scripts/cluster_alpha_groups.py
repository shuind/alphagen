from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alphagen.data.expression import Feature, FeatureType, Ref
from alphagen.models.alpha_cluster import (
    ast_feature_names,
    cluster_ast_features,
    cluster_output_correlations,
    labels_to_members,
)
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.compat import patch_all
from alphagen_qlib.stock_data import StockData
from alphagen_qlib.utils import load_alpha_pool_by_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster alpha pools by output correlation and AST structure.")
    parser.add_argument("--runs-root", required=True, type=str)
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--provider-uri", type=str, default="")
    parser.add_argument("--market", type=str, default="csi300")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--step-mode", type=str, default="all", choices=["all", "list", "range"])
    parser.add_argument("--step-list", type=str, default="")
    parser.add_argument("--step-min", type=int, default=None)
    parser.add_argument("--step-max", type=int, default=None)
    parser.add_argument("--step-stride", type=int, default=1)
    parser.add_argument("--cluster-split", type=str, default="train", choices=["train", "valid", "test", "custom"])
    parser.add_argument("--cluster-start", type=str, default="")
    parser.add_argument("--cluster-end", type=str, default="")
    parser.add_argument("--corr-threshold", type=float, default=0.8)
    parser.add_argument("--ast-threshold", type=float, default=0.9)
    parser.add_argument("--max-backtrack-days", type=int, default=100)
    parser.add_argument("--max-future-days", type=int, default=30)
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_run_path(runs_root: Path, run_id: str) -> Optional[Path]:
    for candidate in (runs_root / run_id, runs_root / "runs" / run_id):
        if candidate.is_dir():
            return candidate
    return None


def _resolve_ckpt_dir(runs_root: Path, run_path: Path, run_id: str) -> Optional[Path]:
    run_meta = _read_json(run_path / "run_meta.json")
    candidates: List[Path] = []
    meta_ckpt_dir = run_meta.get("ckpt_run_dir")
    if isinstance(meta_ckpt_dir, str) and meta_ckpt_dir:
        candidates.append(Path(meta_ckpt_dir))
    candidates.extend(
        [
            run_path / "checkpoints" / run_id,
            run_path.parent / "checkpoints" / run_id,
            runs_root / "checkpoints" / run_id,
        ]
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _list_run_ids(runs_root: Path) -> List[str]:
    nested = runs_root / "runs"
    if nested.is_dir():
        return sorted([p.name for p in nested.iterdir() if p.is_dir()])
    result: List[str] = []
    for p in runs_root.iterdir():
        if not p.is_dir():
            continue
        if (p / "run_meta.json").is_file():
            result.append(p.name)
    return sorted(result)


def _list_step_pools(ckpt_dir: Path) -> List[Tuple[int, Path]]:
    pools: List[Tuple[int, Path]] = []
    for path in ckpt_dir.glob("*_steps_pool.json"):
        stem = path.name.split("_steps_pool.json")[0]
        try:
            step = int(stem)
        except ValueError:
            continue
        pools.append((step, path))
    pools.sort(key=lambda x: x[0])
    return pools


def _parse_step_list(value: str) -> List[int]:
    items: List[int] = []
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            items.append(int(token))
        except ValueError:
            continue
    return sorted(set(items))


def _select_step_pools(
    step_pools: List[Tuple[int, Path]],
    step_mode: str,
    step_list: str,
    step_min: Optional[int],
    step_max: Optional[int],
    step_stride: int,
) -> List[Tuple[int, Path]]:
    if not step_pools:
        return []
    stride = max(1, int(step_stride))
    if step_mode == "list":
        selected = set(_parse_step_list(step_list))
        return [(step, path) for step, path in step_pools if step in selected]
    if step_mode == "range":
        lower = step_min if step_min is not None else step_pools[0][0]
        upper = step_max if step_max is not None else step_pools[-1][0]
        if lower > upper:
            lower, upper = upper, lower
        filtered = [(step, path) for step, path in step_pools if lower <= step <= upper]
        selected = [item for idx, item in enumerate(filtered) if idx % stride == 0]
        if filtered and selected and selected[-1][0] != filtered[-1][0]:
            selected.append(filtered[-1])
        return selected
    if stride == 1:
        return list(step_pools)
    selected = [item for idx, item in enumerate(step_pools) if idx % stride == 0]
    if selected and selected[-1][0] != step_pools[-1][0]:
        selected.append(step_pools[-1])
    return selected


def _resolve_cluster_window(args: argparse.Namespace, run_meta: Dict[str, Any]) -> Tuple[str, str, str]:
    if args.cluster_split == "custom":
        if not args.cluster_start or not args.cluster_end:
            raise ValueError("custom cluster window requires --cluster-start and --cluster-end")
        return args.cluster_start, args.cluster_end, "custom"
    key = args.cluster_split
    start = run_meta.get(f"{key}_start_time")
    end = run_meta.get(f"{key}_end_time")
    if isinstance(start, str) and isinstance(end, str) and start and end:
        return start, end, f"run_meta:{key}"
    defaults = {
        "train": ("2010-01-01", "2019-12-31"),
        "valid": ("2020-01-01", "2020-12-31"),
        "test": ("2021-01-01", "2022-12-31"),
    }
    start, end = defaults[key]
    return start, end, f"default:{key}"


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _build_calculator(
    provider_uri: str,
    market: str,
    device: torch.device,
    start_time: str,
    end_time: str,
    max_backtrack_days: int,
    max_future_days: int,
) -> QLibStockDataCalculator:
    patch_all(provider_uri)
    import qlib
    from qlib.constant import REG_CN

    qlib.init(provider_uri=provider_uri, region=REG_CN)
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1
    data = StockData(
        instrument=market,
        start_time=start_time,
        end_time=end_time,
        max_backtrack_days=max_backtrack_days,
        max_future_days=max_future_days,
        device=device,
    )
    return QLibStockDataCalculator(data, target)


def _single_ic_safe(calculator: QLibStockDataCalculator, expr) -> float:
    try:
        return float(calculator.calc_single_IC_ret(expr))
    except Exception:
        return float("nan")


def _cluster_summary_rows(
    step: int,
    cluster_type: str,
    labels: List[int],
    similarity: np.ndarray,
    exprs: List,
    weights: List[float],
    single_ics: List[float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    members = labels_to_members(labels)
    for cluster_id, idxs in sorted(members.items(), key=lambda x: x[0]):
        abs_weights = [abs(float(weights[i])) for i in idxs]
        rep_idx = idxs[int(np.argmax(abs_weights))] if idxs else -1
        cluster_sim = similarity[np.ix_(idxs, idxs)] if idxs else np.zeros((0, 0), dtype=np.float32)
        rows.append(
            {
                "step": int(step),
                "cluster_type": cluster_type,
                "cluster_id": int(cluster_id),
                "cluster_size": int(len(idxs)),
                "representative_expr": str(exprs[rep_idx]) if rep_idx >= 0 else "",
                "representative_weight": float(weights[rep_idx]) if rep_idx >= 0 else math.nan,
                "mean_abs_weight": float(np.mean(abs_weights)) if abs_weights else math.nan,
                "mean_single_ic": float(np.nanmean([single_ics[i] for i in idxs])) if idxs else math.nan,
                "mean_similarity": float(np.nanmean(cluster_sim)) if cluster_sim.size > 0 else math.nan,
            }
        )
    return rows


def _cluster_label_rows(
    step: int,
    exprs: List,
    weights: List[float],
    single_ics: List[float],
    corr_labels: List[int],
    ast_labels: List[int],
) -> List[Dict[str, Any]]:
    corr_members = labels_to_members(corr_labels)
    ast_members = labels_to_members(ast_labels)
    rows: List[Dict[str, Any]] = []
    for idx, expr in enumerate(exprs):
        corr_id = int(corr_labels[idx]) if idx < len(corr_labels) else -1
        ast_id = int(ast_labels[idx]) if idx < len(ast_labels) else -1
        rows.append(
            {
                "step": int(step),
                "expr_idx": int(idx),
                "expr": str(expr),
                "weight": float(weights[idx]),
                "single_ic": float(single_ics[idx]),
                "corr_cluster_id": corr_id,
                "corr_cluster_size": int(len(corr_members.get(corr_id, []))),
                "ast_cluster_id": ast_id,
                "ast_cluster_size": int(len(ast_members.get(ast_id, []))),
            }
        )
    return rows


def _plot_cluster_sizes(summary_df: pd.DataFrame, cluster_type: str, out_path: Path) -> None:
    subset = summary_df[summary_df["cluster_type"] == cluster_type]
    if subset.empty:
        return
    plt.figure(figsize=(8, 4))
    plt.bar(subset["cluster_id"].astype(str), subset["cluster_size"].astype(float))
    plt.title(f"{cluster_type} cluster sizes")
    plt.xlabel("cluster_id")
    plt.ylabel("cluster_size")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _plot_similarity_heatmap(similarity: np.ndarray, out_path: Path, title: str) -> None:
    if similarity.size == 0:
        return
    plt.figure(figsize=(6, 5))
    plt.imshow(similarity, cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _evaluate_run(args: argparse.Namespace, runs_root: Path, run_id: str) -> Dict[str, Any]:
    run_path = _resolve_run_path(runs_root, run_id)
    if run_path is None:
        raise FileNotFoundError(f"run not found: {run_id}")
    ckpt_dir = _resolve_ckpt_dir(runs_root, run_path, run_id)
    if ckpt_dir is None:
        raise FileNotFoundError(f"checkpoint dir not found for run: {run_id}")
    run_meta = _read_json(run_path / "run_meta.json")
    provider_uri = args.provider_uri or str(run_meta.get("provider_uri", "")).strip()
    if not provider_uri:
        raise ValueError("provider_uri is required")
    step_pools = _select_step_pools(
        _list_step_pools(ckpt_dir),
        args.step_mode,
        args.step_list,
        args.step_min,
        args.step_max,
        args.step_stride,
    )
    if not step_pools:
        raise ValueError(f"no step pools selected for run: {run_id}")

    start_time, end_time, window_source = _resolve_cluster_window(args, run_meta)
    calculator = _build_calculator(
        provider_uri=provider_uri,
        market=args.market or str(run_meta.get("market", "csi300")),
        device=_resolve_device(args.device),
        start_time=start_time,
        end_time=end_time,
        max_backtrack_days=args.max_backtrack_days,
        max_future_days=args.max_future_days,
    )

    label_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    latest_step = -1
    latest_corr_similarity = np.zeros((0, 0), dtype=np.float32)

    for step, pool_path in step_pools:
        exprs, weights = load_alpha_pool_by_path(str(pool_path))
        single_ics = [_single_ic_safe(calculator, expr) for expr in exprs]
        corr_labels, corr_similarity = cluster_output_correlations(exprs, calculator, args.corr_threshold)
        ast_labels, ast_similarity, _ = cluster_ast_features(exprs, args.ast_threshold)
        label_rows.extend(_cluster_label_rows(step, exprs, weights, single_ics, corr_labels, ast_labels))
        summary_rows.extend(_cluster_summary_rows(step, "corr", corr_labels, corr_similarity, exprs, weights, single_ics))
        summary_rows.extend(_cluster_summary_rows(step, "ast", ast_labels, ast_similarity, exprs, weights, single_ics))
        if step >= latest_step:
            latest_step = int(step)
            latest_corr_similarity = corr_similarity

    output_dir = run_path / "cluster_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_df = pd.DataFrame(label_rows)
    summary_df = pd.DataFrame(summary_rows)
    labels_path = output_dir / "step_alpha_cluster_labels.csv"
    summary_path = output_dir / "step_cluster_summary.csv"
    labels_df.to_csv(labels_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    if not summary_df.empty:
        latest_summary = summary_df[summary_df["step"] == latest_step]
        _plot_cluster_sizes(latest_summary, "corr", output_dir / "corr_cluster_size_bar.png")
        _plot_cluster_sizes(latest_summary, "ast", output_dir / "ast_cluster_size_bar.png")
    _plot_similarity_heatmap(latest_corr_similarity, output_dir / "corr_cluster_heatmap.png", f"corr similarity heatmap (step={latest_step})")

    manifest = {
        "run_id": run_id,
        "run_path": str(run_path),
        "ckpt_dir": str(ckpt_dir),
        "provider_uri": provider_uri,
        "market": args.market or str(run_meta.get("market", "csi300")),
        "cluster_split": args.cluster_split,
        "cluster_start": start_time,
        "cluster_end": end_time,
        "cluster_window_source": window_source,
        "corr_threshold": float(args.corr_threshold),
        "ast_threshold": float(args.ast_threshold),
        "step_mode": args.step_mode,
        "step_list": args.step_list,
        "step_min": args.step_min,
        "step_max": args.step_max,
        "step_stride": int(args.step_stride),
        "step_count": int(len(step_pools)),
        "ast_feature_names": ast_feature_names(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest_path = output_dir / "cluster_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "labels_csv": str(labels_path),
        "summary_csv": str(summary_path),
        "manifest_json": str(manifest_path),
    }


def main() -> None:
    args = _parse_args()
    runs_root = Path(args.runs_root)
    run_ids = [args.run_id] if args.run_id else _list_run_ids(runs_root)
    results: List[Dict[str, Any]] = []
    failed: List[Dict[str, str]] = []
    for run_id in run_ids:
        print(f"[start] clustering run={run_id}")
        try:
            results.append(_evaluate_run(args, runs_root, run_id))
            print(f"[done] run={run_id}")
        except Exception as exc:
            failed.append({"run_id": run_id, "error": str(exc)})
            print(f"[fail] run={run_id}: {exc}")
    print(json.dumps({"results": results, "failed": failed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
