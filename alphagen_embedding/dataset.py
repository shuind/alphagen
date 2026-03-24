from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from alphagen.data.expression import *  # noqa: F401,F403
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen_qlib.compat import patch_all
from alphagen_qlib.stock_data import FeatureType, StockData


FEATURE_NAME_MAP = {name.lower(): feature for name, feature in FeatureType.__members__.items()}
POOL_FILE_PATTERN = re.compile(r"(?P<step>\d+)_steps_pool\.json$")
FEATURE_TOKEN_PATTERN = re.compile(r"\$([a-zA-Z_][a-zA-Z0-9_]*)")


def build_default_target(horizon: int = 20) -> Expression:
    close = Feature(FeatureType.CLOSE)
    return Ref(close, -horizon) / close - 1


def _safe_eval_env() -> Dict[str, object]:
    env: Dict[str, object] = {"Feature": Feature, "FeatureType": FeatureType, "Constant": Constant, "DeltaTime": DeltaTime}
    for name, obj in globals().items():
        if name.startswith("_"):
            continue
        if isinstance(obj, type) and issubclass(obj, Expression):
            env[name] = obj
    return env


def parse_expression_text(expr_text: str) -> Expression:
    def _replace_feature(match: re.Match[str]) -> str:
        feature_name = match.group(1).lower()
        if feature_name not in FEATURE_NAME_MAP:
            raise ValueError(f"Unknown feature token: {match.group(0)}")
        return f"Feature(FeatureType.{FEATURE_NAME_MAP[feature_name].name})"

    normalized = FEATURE_TOKEN_PATTERN.sub(_replace_feature, expr_text)
    expr = eval(normalized, {"__builtins__": {}}, _safe_eval_env())
    if not isinstance(expr, Expression):
        raise TypeError(f"Expression text did not produce Expression: {expr_text}")
    return expr


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict json at {path}")
    return data


def load_alpha_source_file(alpha_source_file: str | Path) -> List[Dict[str, object]]:
    path = Path(alpha_source_file)
    payload = _load_json(path)
    raw_alphas = payload.get("alphas", [])
    if not isinstance(raw_alphas, list):
        raise ValueError(f"'alphas' must be a list in {path}")
    result: List[Dict[str, object]] = []
    for idx, item in enumerate(raw_alphas):
        if isinstance(item, str):
            result.append(
                {
                    "run_id": "seed_file",
                    "pool_path": str(path),
                    "expr_text": item,
                    "weight": 0.0,
                    "abs_weight": 0.0,
                    "test_rankic": float("-inf"),
                    "diversity_score": float("-inf"),
                    "order": idx,
                }
            )
            continue
        if not isinstance(item, dict):
            raise ValueError(f"Alpha entry must be str or dict, got {type(item)}")
        expr_text = item.get("expr_text") or item.get("expr")
        if not isinstance(expr_text, str):
            raise ValueError(f"Missing expr_text in alpha entry #{idx}")
        result.append(
            {
                "run_id": str(item.get("run_id", "seed_file")),
                "pool_path": str(item.get("pool_path", path)),
                "expr_text": expr_text,
                "weight": float(item.get("weight", 0.0)),
                "abs_weight": abs(float(item.get("weight", 0.0))),
                "test_rankic": float(item.get("test_rankic", float("-inf"))),
                "diversity_score": float(item.get("diversity_score", float("-inf"))),
                "order": int(item.get("order", idx)),
            }
        )
    return result


def _scan_latest_pool_files(checkpoint_root: Path) -> List[Path]:
    latest_by_run: Dict[str, tuple[int, Path]] = {}
    if not checkpoint_root.is_dir():
        return []
    for path in checkpoint_root.rglob("*_steps_pool.json"):
        match = POOL_FILE_PATTERN.search(path.name)
        if match is None:
            continue
        step = int(match.group("step"))
        run_id = path.parent.name
        prev = latest_by_run.get(run_id)
        if prev is None or step > prev[0]:
            latest_by_run[run_id] = (step, path)
    return [info[1] for info in sorted(latest_by_run.values(), key=lambda item: item[1].parent.name)]


def _load_run_summaries(runs_root: Path) -> Dict[str, Dict[str, object]]:
    summaries: Dict[str, Dict[str, object]] = {}
    if not runs_root.is_dir():
        return summaries
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "run_meta.json"
        summary_path = run_dir / "summary_metrics.json"
        meta = _load_json(meta_path) if meta_path.is_file() else {}
        summary = _load_json(summary_path) if summary_path.is_file() else {}
        merged = {}
        merged.update(meta)
        merged.update(summary)
        summaries[run_dir.name] = merged
    return summaries


def collect_alpha_candidates(
    checkpoint_root: str | Path,
    runs_root: str | Path,
    min_alpha: int = 20,
    max_alpha: int = 50,
) -> List[Dict[str, object]]:
    checkpoint_root = Path(checkpoint_root)
    runs_root = Path(runs_root)
    run_summaries = _load_run_summaries(runs_root)
    latest_pool_files = _scan_latest_pool_files(checkpoint_root)

    candidates: List[Dict[str, object]] = []
    for pool_path in latest_pool_files:
        run_id = pool_path.parent.name
        pool_data = _load_json(pool_path)
        exprs = pool_data.get("exprs", [])
        weights = pool_data.get("weights", [])
        if not isinstance(exprs, list) or not isinstance(weights, list):
            continue
        run_summary = run_summaries.get(run_id, {})
        for idx, (expr_text, weight) in enumerate(zip(exprs, weights)):
            candidates.append(
                {
                    "run_id": run_id,
                    "pool_path": str(pool_path),
                    "expr_text": str(expr_text),
                    "weight": float(weight),
                    "abs_weight": abs(float(weight)),
                    "test_rankic": float(run_summary.get("test_rankic", float("-inf"))),
                    "diversity_score": float(run_summary.get("diversity_score", float("-inf"))),
                    "order": idx,
                }
            )

    candidates.sort(
        key=lambda row: (
            row["test_rankic"],
            row["diversity_score"],
            row["abs_weight"],
            row["run_id"],
            -row["order"],
        ),
        reverse=True,
    )

    deduped: List[Dict[str, object]] = []
    seen = set()
    for row in candidates:
        expr_text = row["expr_text"]
        if expr_text in seen:
            continue
        seen.add(expr_text)
        deduped.append(row)
        if len(deduped) >= max_alpha:
            break

    if len(deduped) < min_alpha:
        raise ValueError(f"Only collected {len(deduped)} unique alpha expressions, need at least {min_alpha}.")
    return deduped


def _prepare_environment(provider_uri: str) -> None:
    os.environ["QLIB_PROVIDER_URI"] = provider_uri
    patch_all(provider_uri)


def _daily_top_bottom_return(alpha_panel: Tensor, target_panel: Tensor, top_frac: float = 0.2) -> Tensor:
    days, stocks = alpha_panel.shape
    k = max(1, int(stocks * top_frac))
    returns = torch.zeros(days, dtype=alpha_panel.dtype)
    for day in range(days):
        alpha_day = alpha_panel[day]
        target_day = target_panel[day]
        top_idx = torch.topk(alpha_day, k=k, largest=True).indices
        bottom_idx = torch.topk(alpha_day, k=k, largest=False).indices
        returns[day] = target_day[top_idx].mean() - target_day[bottom_idx].mean()
    return returns


def _split_ranges(num_samples: int, train_ratio: float = 0.6, valid_ratio: float = 0.2) -> Dict[str, List[int]]:
    train_end = int(num_samples * train_ratio)
    valid_end = int(num_samples * (train_ratio + valid_ratio))
    return {
        "train": list(range(0, train_end)),
        "valid": list(range(train_end, valid_end)),
        "test": list(range(valid_end, num_samples)),
    }


def build_embedding_dataset(
    checkpoint_root: str | Path,
    runs_root: str | Path,
    provider_uri: str,
    market: str = "csi300",
    start_time: str = "2021-01-01",
    end_time: str = "2022-12-31",
    num_alpha: int = 20,
    stock_limit: int = 100,
    lookback: int = 60,
    horizon: int = 20,
    device: str = "cpu",
    alpha_candidates: Optional[List[Dict[str, object]]] = None,
) -> Dict[str, object]:
    _prepare_environment(provider_uri)
    if alpha_candidates is None:
        alpha_candidates = collect_alpha_candidates(checkpoint_root, runs_root, min_alpha=num_alpha, max_alpha=max(num_alpha, 50))
    selected = alpha_candidates[:num_alpha]
    expr_texts = [row["expr_text"] for row in selected]
    exprs = [parse_expression_text(expr_text) for expr_text in expr_texts]
    stock_data = StockData(
        instrument=market,
        start_time=start_time,
        end_time=end_time,
        device=torch.device(device),
    )

    target_expr = build_default_target(horizon=horizon)
    target_raw = target_expr.evaluate(stock_data)[:, :stock_limit].detach().cpu()
    alpha_values = []
    for expr in exprs:
        alpha_values.append(normalize_by_day(expr.evaluate(stock_data))[:, :stock_limit].detach().cpu())
    alpha_panel = torch.stack(alpha_values, dim=1)  # [days, alpha, stocks]
    days, alpha_count, stocks = alpha_panel.shape

    ic_daily = []
    rankic_daily = []
    return_daily = []
    turnover_daily = []
    for alpha_idx in range(alpha_count):
        alpha_day_panel = alpha_panel[:, alpha_idx, :]
        ic_daily.append(torch.nan_to_num(batch_pearsonr(alpha_day_panel, target_raw), nan=0.0, posinf=0.0, neginf=0.0))
        rankic_daily.append(torch.nan_to_num(batch_spearmanr(alpha_day_panel, target_raw), nan=0.0, posinf=0.0, neginf=0.0))
        return_daily.append(torch.nan_to_num(_daily_top_bottom_return(alpha_day_panel, target_raw), nan=0.0, posinf=0.0, neginf=0.0))
        turnover_daily.append(torch.zeros(days, dtype=alpha_day_panel.dtype))

    behavior_daily = torch.stack([torch.stack(ic_daily, dim=1), torch.stack(rankic_daily, dim=1), torch.stack(return_daily, dim=1)], dim=2)
    behavior_daily = torch.nan_to_num(behavior_daily, nan=0.0, posinf=0.0, neginf=0.0)
    turnover_daily_tensor = torch.stack(turnover_daily, dim=1)

    behavior_windows = []
    stat_windows = []
    alpha_cross_sections = []
    target_cross_sections = []
    sample_dates: List[str] = []
    for day_idx in range(lookback, days):
        history = behavior_daily[day_idx - lookback:day_idx]  # [L, alpha, 3]
        history = history.permute(1, 0, 2).contiguous()  # [alpha, L, 3]
        turnover_hist = turnover_daily_tensor[day_idx - lookback:day_idx].permute(1, 0).contiguous()
        mean_ic = history[:, :, 0].mean(dim=1)
        std_ic = history[:, :, 0].std(dim=1, unbiased=False)
        mean_ret = history[:, :, 2].mean(dim=1)
        std_ret = history[:, :, 2].std(dim=1, unbiased=False)
        mean_turnover = turnover_hist.mean(dim=1)
        stats = torch.stack([mean_ic, std_ic, mean_ret, std_ret, mean_turnover], dim=1)
        stats = torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)

        behavior_windows.append(history)
        stat_windows.append(stats)
        alpha_cross_sections.append(alpha_panel[day_idx])
        target_cross_sections.append(target_raw[day_idx])
        sample_dates.append(str(stock_data._dates[stock_data.max_backtrack_days + day_idx]))

    behavior_tensor = torch.stack(behavior_windows, dim=0).float()
    stats_tensor = torch.stack(stat_windows, dim=0).float()
    alpha_tensor = torch.stack(alpha_cross_sections, dim=0).float()
    target_tensor = torch.nan_to_num(torch.stack(target_cross_sections, dim=0).float(), nan=0.0, posinf=0.0, neginf=0.0)
    split_indices = _split_ranges(behavior_tensor.shape[0])

    alpha_summary = []
    for alpha_idx, expr_text in enumerate(expr_texts):
        alpha_summary.append(
            {
                "alpha_id": alpha_idx,
                "expr_text": expr_text,
                "mean_ic": float(behavior_daily[:, alpha_idx, 0].mean().item()),
                "std_ic": float(behavior_daily[:, alpha_idx, 0].std(unbiased=False).item()),
                "mean_rankic": float(behavior_daily[:, alpha_idx, 1].mean().item()),
                "mean_return": float(behavior_daily[:, alpha_idx, 2].mean().item()),
                "source_run_id": selected[alpha_idx]["run_id"],
                "source_weight": selected[alpha_idx]["weight"],
                "source_test_rankic": selected[alpha_idx]["test_rankic"],
                "source_diversity_score": selected[alpha_idx]["diversity_score"],
            }
        )

    return {
        "behavior": behavior_tensor,
        "stats": stats_tensor,
        "alpha_values": alpha_tensor,
        "targets": target_tensor,
        "split_indices": split_indices,
        "sample_dates": sample_dates,
        "alpha_summary": alpha_summary,
        "expr_texts": expr_texts,
        "stock_ids": [str(x) for x in stock_data._stock_ids[:stock_limit]],
        "config": {
            "market": market,
            "start_time": start_time,
            "end_time": end_time,
            "num_alpha": num_alpha,
            "stock_limit": stock_limit,
            "lookback": lookback,
            "horizon": horizon,
            "provider_uri": provider_uri,
            "alpha_source_mode": "file" if alpha_candidates is not None else "runs",
        },
    }


def save_embedding_dataset(dataset: Dict[str, object], output_dir: str | Path) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "embedding_dataset.pt"
    meta_path = output_dir / "embedding_dataset_meta.json"
    torch.save(dataset, dataset_path)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": dataset["config"],
                "num_samples": int(dataset["behavior"].shape[0]),
                "num_alpha": int(dataset["behavior"].shape[1]),
                "lookback": int(dataset["behavior"].shape[2]),
                "feature_dim": int(dataset["behavior"].shape[3]),
                "stock_limit": int(dataset["alpha_values"].shape[2]),
                "split_sizes": {key: len(value) for key, value in dataset["split_indices"].items()},
                "alpha_summary": dataset["alpha_summary"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return {"dataset_path": str(dataset_path), "meta_path": str(meta_path)}
