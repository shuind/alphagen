from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alphagen.data.expression import Feature, FeatureType, Ref
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.compat import patch_all
from alphagen_qlib.stock_data import StockData
from alphagen_qlib.utils import load_alpha_pool_by_path


@dataclass
class StepPool:
    step: int
    path: Path


@dataclass
class YearCalculatorBuild:
    year: int
    elapsed_sec: float
    status: str
    error: str = ""


@dataclass
class CalculatorCacheEntry:
    calculators: Dict[int, QLibStockDataCalculator]
    year_build_stats: List[YearCalculatorBuild]
    build_total_sec: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate each checkpoint pool by yearly generalization metrics "
            "(pre-train years vs post-train years)."
        )
    )
    parser.add_argument("--runs-root", required=True, type=str)
    parser.add_argument("--run-id", type=str, default="", help="single run id or comma-separated run ids")
    parser.add_argument("--provider-uri", type=str, default="")
    parser.add_argument("--market", type=str, default="csi300")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--step-mode", type=str, default="all", choices=["all", "list", "range"])
    parser.add_argument(
        "--step-list",
        type=str,
        default="",
        help="Comma-separated steps used when step-mode=list, e.g. 2048,4096,8192.",
    )
    parser.add_argument("--step-min", type=int, default=None)
    parser.add_argument("--step-max", type=int, default=None)
    parser.add_argument("--step-stride", type=int, default=1)
    parser.add_argument(
        "--years",
        type=str,
        default="",
        help="Comma-separated explicit years to evaluate, e.g. 2018,2019,2020.",
    )
    parser.add_argument(
        "--pre-years-limit",
        type=int,
        default=0,
        help="If >0, keep only the nearest N pre-train years.",
    )
    parser.add_argument(
        "--post-years-limit",
        type=int,
        default=0,
        help="If >0, keep only the nearest N post-train years.",
    )
    parser.add_argument("--train-start-year", type=int, default=None)
    parser.add_argument("--train-end-year", type=int, default=None)
    parser.add_argument("--max-backtrack-days", type=int, default=100)
    parser.add_argument("--max-future-days", type=int, default=30)
    parser.add_argument("--qlib-n-jobs", type=int, default=1)
    parser.add_argument("--consistency-check", action="store_true")
    parser.add_argument("--consistency-reference", type=str, default="")
    parser.add_argument("--consistency-tolerance", type=float, default=1e-8)
    return parser.parse_args()


def _read_json(path: Path) -> Dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_run_path(runs_root: Path, run_id: str) -> Optional[Path]:
    candidates = [
        runs_root / run_id,
        runs_root / "runs" / run_id,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _resolve_ckpt_dir(runs_root: Path, run_path: Path, run_id: str) -> Optional[Path]:
    run_meta = _read_json(run_path / "run_meta.json")
    meta_ckpt_dir = run_meta.get("ckpt_run_dir")
    candidates: List[Path] = []
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
    nested_runs = runs_root / "runs"
    if nested_runs.is_dir():
        return sorted([p.name for p in nested_runs.iterdir() if p.is_dir()])

    skip_names = {"runs", "checkpoints", "tb_log"}
    run_ids: List[str] = []
    for p in runs_root.iterdir():
        if not p.is_dir() or p.name in skip_names:
            continue
        if (p / "run_meta.json").is_file() or (p / "metrics_step.csv").is_file():
            run_ids.append(p.name)
    return sorted(run_ids)


def _parse_step_list_arg(step_list_arg: str) -> List[int]:
    steps: List[int] = []
    for part in step_list_arg.split(","):
        t = part.strip()
        if not t:
            continue
        try:
            steps.append(int(t))
        except ValueError:
            continue
    return sorted(list(set(steps)))


def _list_step_pools(ckpt_dir: Path) -> List[StepPool]:
    step_pools: List[StepPool] = []
    for path in ckpt_dir.glob("*_steps_pool.json"):
        stem = path.name.split("_steps_pool.json")[0]
        try:
            step = int(stem)
        except ValueError:
            continue
        step_pools.append(StepPool(step=step, path=path))
    step_pools.sort(key=lambda x: x.step)
    return step_pools


def _select_step_pools(
    step_pools: List[StepPool],
    step_mode: str,
    step_list_arg: str,
    step_min: Optional[int],
    step_max: Optional[int],
    step_stride: int,
) -> List[StepPool]:
    if not step_pools:
        return []

    if step_mode == "list":
        selected_steps = set(_parse_step_list_arg(step_list_arg))
        if not selected_steps:
            return []
        return [sp for sp in step_pools if sp.step in selected_steps]

    stride = max(1, step_stride)
    if step_mode == "range":
        lower = step_min if step_min is not None else step_pools[0].step
        upper = step_max if step_max is not None else step_pools[-1].step
        if lower > upper:
            lower, upper = upper, lower
        in_range = [sp for sp in step_pools if lower <= sp.step <= upper]
        if not in_range:
            return []
        selected = [sp for idx, sp in enumerate(in_range) if idx % stride == 0]
        if selected and selected[-1].step != in_range[-1].step:
            selected.append(in_range[-1])
        return selected

    # step_mode == all
    if stride == 1:
        return list(step_pools)
    selected = [sp for idx, sp in enumerate(step_pools) if idx % stride == 0]
    if selected and selected[-1].step != step_pools[-1].step:
        selected.append(step_pools[-1])
    return selected


def _safe_float(value: float) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return float(value)


def _ir_from_tensor(values: torch.Tensor) -> float:
    finite = values[~values.isnan()]
    if finite.numel() == 0:
        return float("nan")
    std = finite.std(unbiased=False).item()
    if std < 1e-12:
        return float("nan")
    return (finite.mean().item()) / std


def _mean_from_tensor(values: torch.Tensor) -> float:
    finite = values[~values.isnan()]
    if finite.numel() == 0:
        return float("nan")
    return finite.mean().item()


def _eval_pool_on_calculator(
    calculator: QLibStockDataCalculator,
    exprs: List,
    weights: List[float],
) -> Dict[str, float]:
    with torch.no_grad():
        signal = calculator.make_ensemble_alpha(exprs, weights)
        target = calculator.target_value
        daily_ic = batch_pearsonr(signal, target)
        daily_rankic = batch_spearmanr(signal, target)

        valid_mask = ~(signal.isnan() | target.isnan())
        n_valid_stocks = valid_mask.sum(dim=1).float()
        n_stocks = valid_mask.shape[1]

    return {
        "year_ic": _mean_from_tensor(daily_ic),
        "year_rankic": _mean_from_tensor(daily_rankic),
        "year_icir": _ir_from_tensor(daily_ic),
        "year_rankicir": _ir_from_tensor(daily_rankic),
        "n_days": float(daily_rankic.shape[0]),
        "n_valid_days_ic": float((~daily_ic.isnan()).sum().item()),
        "n_valid_days_rankic": float((~daily_rankic.isnan()).sum().item()),
        "n_stocks_mean": n_valid_stocks.mean().item(),
        "valid_ratio": (n_valid_stocks / max(1, n_stocks)).mean().item(),
    }


def _agg(series: pd.Series) -> Tuple[float, float, float, int]:
    vals = pd.to_numeric(series, errors="coerce").dropna().astype(float).values
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    return float(np.mean(vals)), float(np.std(vals, ddof=0)), float(np.var(vals, ddof=0)), int(len(vals))


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        return torch.device("cuda:0")
    return torch.device("cpu")


def _resolve_provider_uri(cli_provider: str, run_path: Path) -> str:
    def _non_empty(value: str | None) -> str:
        return (value or "").strip()

    candidates: List[str] = []
    if _non_empty(cli_provider):
        candidates.append(_non_empty(cli_provider))

    for env_name in ["QLIB_PROVIDER_URI", "PROVIDER_URI"]:
        env_val = _non_empty(os.environ.get(env_name, ""))
        if env_val:
            candidates.append(env_val)

    meta = _read_json(run_path / "run_meta.json")
    meta_provider = meta.get("provider_uri")
    if isinstance(meta_provider, str) and _non_empty(meta_provider):
        candidates.append(_non_empty(meta_provider))

    # Legacy fallback for Kaggle environments.
    candidates.append("/kaggle/input/baostock/cn_data_baostock_fwdadj")

    for candidate in candidates:
        p = Path(candidate)
        if p.exists():
            return candidate

    # Return first candidate for downstream error context if none exists.
    return candidates[0]


def _init_qlib(provider_uri: str, n_jobs: int) -> None:
    patch_all(provider_uri)
    import qlib
    from qlib.constant import REG_CN
    from qlib.config import C

    qlib.init(provider_uri=provider_uri, region=REG_CN)
    if n_jobs > 0:
        try:
            C.joblib_n_jobs = n_jobs
        except Exception:
            try:
                C["joblib_n_jobs"] = n_jobs
            except Exception:
                pass


def _collect_eval_years(
    train_start_year: int,
    train_end_year: int,
    max_backtrack_days: int,
    max_future_days: int,
) -> Tuple[List[int], List[int], List[int]]:
    from qlib.data import D

    calendar = pd.to_datetime(D.calendar())
    years = sorted(set(calendar.year))
    valid_years: List[int] = []
    for year in years:
        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp(f"{year}-12-31")
        start_idx = int(calendar.searchsorted(start, side="left"))
        end_idx = int(calendar.searchsorted(end, side="right")) - 1
        if end_idx <= start_idx:
            continue
        if start_idx < max_backtrack_days:
            continue
        if end_idx + max_future_days >= len(calendar):
            continue
        valid_years.append(year)

    pre_years = [y for y in valid_years if y < train_start_year]
    post_years = [y for y in valid_years if y > train_end_year]
    eval_years = sorted(pre_years + post_years)
    return pre_years, post_years, eval_years


def _apply_year_filters(
    pre_years: List[int],
    post_years: List[int],
    years_arg: str,
    pre_limit: int,
    post_limit: int,
) -> Tuple[List[int], List[int], List[int]]:
    selected_pre = list(pre_years)
    selected_post = list(post_years)

    if years_arg.strip():
        parsed = []
        for part in years_arg.split(","):
            t = part.strip()
            if not t:
                continue
            try:
                parsed.append(int(t))
            except ValueError:
                continue
        keep = set(parsed)
        selected_pre = [y for y in selected_pre if y in keep]
        selected_post = [y for y in selected_post if y in keep]

    if pre_limit > 0 and len(selected_pre) > pre_limit:
        # Keep years closest to train start (latest pre years).
        selected_pre = selected_pre[-pre_limit:]
    if post_limit > 0 and len(selected_post) > post_limit:
        # Keep years closest to train end (earliest post years).
        selected_post = selected_post[:post_limit]

    return selected_pre, selected_post, sorted(selected_pre + selected_post)


def _build_year_calculators(
    market: str,
    eval_years: Iterable[int],
    device: torch.device,
    max_backtrack_days: int,
    max_future_days: int,
) -> Tuple[Dict[int, QLibStockDataCalculator], List[YearCalculatorBuild]]:
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1
    calculators: Dict[int, QLibStockDataCalculator] = {}
    build_stats: List[YearCalculatorBuild] = []
    years = list(eval_years)
    total = len(years)
    for idx, year in enumerate(years, start=1):
        start_time = f"{year}-01-01"
        end_time = f"{year}-12-31"
        t0 = time.perf_counter()
        try:
            print(f"[year] building calculator {idx}/{total}: {year}")
            stock_data = StockData(
                instrument=market,
                start_time=start_time,
                end_time=end_time,
                max_backtrack_days=max_backtrack_days,
                max_future_days=max_future_days,
                device=device,
            )
            if stock_data.n_days <= 0:
                build_stats.append(
                    YearCalculatorBuild(
                        year=year,
                        elapsed_sec=max(0.0, time.perf_counter() - t0),
                        status="skip",
                        error="n_days<=0",
                    )
                )
                continue
            calculator = QLibStockDataCalculator(stock_data, target)
            with torch.no_grad():
                quality_signal = calculator._calc_alpha(close)
                quality_target = calculator.target_value
                daily_rankic = batch_spearmanr(quality_signal, quality_target)
                valid_mask = ~(quality_signal.isnan() | quality_target.isnan())
                n_valid_stocks = valid_mask.sum(dim=1).float()
                valid_ratio = (n_valid_stocks / max(1, valid_mask.shape[1])).mean().item()
                n_valid_days = int((~daily_rankic.isnan()).sum().item())
            if n_valid_days <= 0 or valid_ratio <= 0:
                print(f"[year] skip {year}: no valid daily samples")
                build_stats.append(
                    YearCalculatorBuild(
                        year=year,
                        elapsed_sec=max(0.0, time.perf_counter() - t0),
                        status="skip",
                        error="no_valid_daily_samples",
                    )
                )
                continue
            calculators[year] = calculator
            build_stats.append(
                YearCalculatorBuild(
                    year=year,
                    elapsed_sec=max(0.0, time.perf_counter() - t0),
                    status="ok",
                    error="",
                )
            )
        except Exception as exc:
            print(f"[warn] skip year={year}: {exc}")
            build_stats.append(
                YearCalculatorBuild(
                    year=year,
                    elapsed_sec=max(0.0, time.perf_counter() - t0),
                    status="error",
                    error=str(exc),
                )
            )
    return calculators, build_stats


def _calc_cache_key(
    provider_uri: str,
    market: str,
    device: torch.device,
    max_backtrack_days: int,
    max_future_days: int,
    eval_years: Iterable[int],
) -> Tuple[str, str, str, int, int, Tuple[int, ...]]:
    return (
        provider_uri,
        market,
        str(device),
        int(max_backtrack_days),
        int(max_future_days),
        tuple(sorted(list(eval_years))),
    )


def _extract_year_from_value(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        if 1900 <= int(value) <= 2200:
            return int(value)
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.isdigit() and len(text) == 4:
            year = int(text)
            if 1900 <= year <= 2200:
                return year
            return None
        try:
            return int(pd.Timestamp(text).year)
        except Exception:
            return None
    return None


def _resolve_train_window_years(
    args: argparse.Namespace,
    run_meta: Dict[str, Any],
) -> Tuple[int, int, str, List[str]]:
    warnings: List[str] = []
    default_start, default_end = 2010, 2019

    if args.train_start_year is not None and args.train_end_year is not None:
        return int(args.train_start_year), int(args.train_end_year), "cli", warnings

    data_windows = run_meta.get("data_windows", {}) if isinstance(run_meta.get("data_windows", {}), dict) else {}
    train_window = data_windows.get("train", {}) if isinstance(data_windows.get("train", {}), dict) else {}
    alt_train_window = run_meta.get("train_window", {}) if isinstance(run_meta.get("train_window", {}), dict) else {}

    meta_start = (
        _extract_year_from_value(run_meta.get("train_start_year"))
        or _extract_year_from_value(run_meta.get("train_start_time"))
        or _extract_year_from_value(train_window.get("start"))
        or _extract_year_from_value(train_window.get("start_time"))
        or _extract_year_from_value(alt_train_window.get("start"))
        or _extract_year_from_value(alt_train_window.get("start_time"))
    )
    meta_end = (
        _extract_year_from_value(run_meta.get("train_end_year"))
        or _extract_year_from_value(run_meta.get("train_end_time"))
        or _extract_year_from_value(train_window.get("end"))
        or _extract_year_from_value(train_window.get("end_time"))
        or _extract_year_from_value(alt_train_window.get("end"))
        or _extract_year_from_value(alt_train_window.get("end_time"))
    )

    cli_start = int(args.train_start_year) if args.train_start_year is not None else None
    cli_end = int(args.train_end_year) if args.train_end_year is not None else None

    if cli_start is not None and cli_end is None:
        start = cli_start
        end = meta_end if meta_end is not None else default_end
        source = "mixed_cli+meta" if meta_end is not None else "mixed_cli+default"
        if meta_end is None:
            warnings.append(f"train_end_year missing in run_meta; fallback to default {default_end}")
        return start, end, source, warnings

    if cli_end is not None and cli_start is None:
        end = cli_end
        start = meta_start if meta_start is not None else default_start
        source = "mixed_cli+meta" if meta_start is not None else "mixed_cli+default"
        if meta_start is None:
            warnings.append(f"train_start_year missing in run_meta; fallback to default {default_start}")
        return start, end, source, warnings

    if meta_start is not None and meta_end is not None:
        return meta_start, meta_end, "run_meta", warnings

    if meta_start is None:
        warnings.append(f"train_start_year missing in run_meta; fallback to default {default_start}")
    if meta_end is None:
        warnings.append(f"train_end_year missing in run_meta; fallback to default {default_end}")
    return default_start, default_end, "default", warnings


def _resolve_git_revision() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        return out.strip()
    except Exception:
        return "unknown"


def _extract_consistency_keys(summary: Dict[str, Any]) -> Dict[str, Any]:
    best_metrics = summary.get("best_metrics", {}) if isinstance(summary.get("best_metrics", {}), dict) else {}
    return {
        "best_step_by_post_rankic": summary.get("best_step_by_post_rankic"),
        "post_mean_rankic": best_metrics.get("post_mean_rankic"),
        "post_var_rankic": best_metrics.get("post_var_rankic"),
        "generalization_gap": best_metrics.get("generalization_gap"),
        "stability_score": best_metrics.get("stability_score"),
    }


def _compare_consistency(
    reference: Dict[str, Any],
    current: Dict[str, Any],
    tolerance: float,
) -> Dict[str, Any]:
    ref_keys = _extract_consistency_keys(reference)
    cur_keys = _extract_consistency_keys(current)
    diffs: List[Dict[str, Any]] = []

    for key, ref_value in ref_keys.items():
        cur_value = cur_keys.get(key)
        if isinstance(ref_value, (int, np.integer)) and isinstance(cur_value, (int, np.integer)):
            if int(ref_value) != int(cur_value):
                diffs.append({"key": key, "reference": int(ref_value), "current": int(cur_value), "delta": None})
            continue

        try:
            ref_num = float(ref_value)
            cur_num = float(cur_value)
            if math.isnan(ref_num) and math.isnan(cur_num):
                continue
            delta = abs(cur_num - ref_num)
            if math.isnan(delta) or delta > tolerance:
                diffs.append({"key": key, "reference": ref_num, "current": cur_num, "delta": delta})
            continue
        except Exception:
            if ref_value != cur_value:
                diffs.append({"key": key, "reference": ref_value, "current": cur_value, "delta": None})

    return {
        "passed": len(diffs) == 0,
        "differences": diffs,
        "tolerance": tolerance,
        "reference_keys": ref_keys,
        "current_keys": cur_keys,
    }


def _evaluate_run(
    run_id: str,
    runs_root: Path,
    args: argparse.Namespace,
    calculator_cache: Optional[Dict[Tuple[str, str, str, int, int, Tuple[int, ...]], CalculatorCacheEntry]] = None,
) -> Dict:
    run_path = _resolve_run_path(runs_root, run_id)
    if run_path is None:
        raise FileNotFoundError(f"run not found: {run_id}")
    ckpt_dir = _resolve_ckpt_dir(runs_root, run_path, run_id)
    if ckpt_dir is None:
        raise FileNotFoundError(f"checkpoint dir not found for run: {run_id}")
    run_meta = _read_json(run_path / "run_meta.json")

    provider_uri = _resolve_provider_uri(args.provider_uri, run_path)
    _init_qlib(provider_uri, args.qlib_n_jobs)
    device = _resolve_device(args.device)
    train_start_year, train_end_year, train_window_source, window_warnings = _resolve_train_window_years(args, run_meta)

    for warning in window_warnings:
        print(f"[warn] {warning}")
    print(f"[train-window] source={train_window_source} start={train_start_year} end={train_end_year}")

    run_started_at = datetime.now()
    run_eval_t0 = time.perf_counter()
    pre_years, post_years, eval_years = _collect_eval_years(
        train_start_year=train_start_year,
        train_end_year=train_end_year,
        max_backtrack_days=args.max_backtrack_days,
        max_future_days=args.max_future_days,
    )
    pre_years, post_years, eval_years = _apply_year_filters(
        pre_years=pre_years,
        post_years=post_years,
        years_arg=args.years,
        pre_limit=max(0, args.pre_years_limit),
        post_limit=max(0, args.post_years_limit),
    )
    if not eval_years:
        raise RuntimeError(
            "no evaluable pre/post years found from calendar; "
            "check provider data span and train-year config"
        )
    print(
        f"[year] selected pre={pre_years} post={post_years} "
        f"(total={len(eval_years)})"
    )

    cache_key = _calc_cache_key(
        provider_uri=provider_uri,
        market=args.market,
        device=device,
        max_backtrack_days=args.max_backtrack_days,
        max_future_days=args.max_future_days,
        eval_years=eval_years,
    )
    calculators: Dict[int, QLibStockDataCalculator]
    year_build_stats: List[YearCalculatorBuild]
    calc_build_total_sec: float
    calc_build_cache_hit = False
    if calculator_cache is not None and cache_key in calculator_cache:
        cache_entry = calculator_cache[cache_key]
        calculators = cache_entry.calculators
        year_build_stats = cache_entry.year_build_stats
        calc_build_total_sec = 0.0
        calc_build_cache_hit = True
        print(f"[year] reuse cached calculators: years={len(calculators)}")
    else:
        calc_build_t0 = time.perf_counter()
        calculators, year_build_stats = _build_year_calculators(
            market=args.market,
            eval_years=eval_years,
            device=device,
            max_backtrack_days=args.max_backtrack_days,
            max_future_days=args.max_future_days,
        )
        calc_build_total_sec = max(0.0, time.perf_counter() - calc_build_t0)
        if calculator_cache is not None:
            calculator_cache[cache_key] = CalculatorCacheEntry(
                calculators=calculators,
                year_build_stats=year_build_stats,
                build_total_sec=calc_build_total_sec,
            )

    if not calculators:
        raise RuntimeError("no yearly calculators built successfully")
    effective_years = sorted(list(calculators.keys()))
    effective_pre_years = [y for y in effective_years if y < train_start_year]
    effective_post_years = [y for y in effective_years if y > train_end_year]

    all_step_pools = _list_step_pools(ckpt_dir)
    step_pools = _select_step_pools(
        all_step_pools,
        step_mode=args.step_mode,
        step_list_arg=args.step_list,
        step_min=args.step_min,
        step_max=args.step_max,
        step_stride=args.step_stride,
    )
    if not step_pools:
        raise FileNotFoundError(
            f"no matching *_steps_pool.json under {ckpt_dir} for step selector "
            f"(mode={args.step_mode}, list='{args.step_list}', min={args.step_min}, max={args.step_max}, stride={args.step_stride})"
        )

    year_rows: List[Dict] = []
    stability_rows: List[Dict] = []
    step_timing_rows: List[Dict[str, Any]] = []
    step_year_failures: List[Dict[str, Any]] = []
    consistency_reference_payload: Dict[str, Any] = {}

    for idx, step_pool in enumerate(step_pools, start=1):
        step_t0 = time.perf_counter()
        print(f"[step] evaluating checkpoint {idx}/{len(step_pools)}: step={step_pool.step}")
        try:
            exprs, weights = load_alpha_pool_by_path(str(step_pool.path))
        except Exception as exc:
            step_year_failures.append(
                {
                    "step": step_pool.step,
                    "year": None,
                    "stage": "load_pool",
                    "error": str(exc),
                }
            )
            step_timing_rows.append(
                {
                    "step": step_pool.step,
                    "elapsed_sec": max(0.0, time.perf_counter() - step_t0),
                    "evaluated_year_count": 0,
                    "failed_year_count": 1,
                }
            )
            print(f"[warn] step={step_pool.step} load failed: {exc}")
            continue
        if len(exprs) == 0:
            step_timing_rows.append(
                {
                    "step": step_pool.step,
                    "elapsed_sec": max(0.0, time.perf_counter() - step_t0),
                    "evaluated_year_count": 0,
                    "failed_year_count": 0,
                }
            )
            continue
        if len(exprs) != len(weights):
            m = min(len(exprs), len(weights))
            exprs, weights = exprs[:m], weights[:m]

        this_step_rows: List[Dict] = []
        this_step_failures = 0
        for year, calculator in calculators.items():
            year_t0 = time.perf_counter()
            try:
                metrics = _eval_pool_on_calculator(calculator, exprs, weights)
                period = "pre" if year < train_start_year else "post"
                row = {
                    "run_id": run_id,
                    "step": step_pool.step,
                    "pool_json": str(step_pool.path),
                    "pool_size": len(exprs),
                    "year": year,
                    "period": period,
                    "eval_sec": max(0.0, time.perf_counter() - year_t0),
                    **metrics,
                }
                year_rows.append(row)
                this_step_rows.append(row)
            except Exception as exc:
                this_step_failures += 1
                fail_payload = {
                    "step": step_pool.step,
                    "year": year,
                    "stage": "eval_year",
                    "error": str(exc),
                    "elapsed_sec": max(0.0, time.perf_counter() - year_t0),
                }
                step_year_failures.append(fail_payload)
                print(f"[warn] step={step_pool.step} year={year} eval failed: {exc}")

        step_elapsed = max(0.0, time.perf_counter() - step_t0)
        step_timing_rows.append(
            {
                "step": step_pool.step,
                "elapsed_sec": step_elapsed,
                "evaluated_year_count": len(this_step_rows),
                "failed_year_count": this_step_failures,
            }
        )
        print(
            f"[step] done step={step_pool.step} "
            f"evaluated_years={len(this_step_rows)} failed_years={this_step_failures} "
            f"elapsed={step_elapsed:.2f}s"
        )
        if not this_step_rows:
            continue

        step_df = pd.DataFrame(this_step_rows)
        pre_df = step_df[step_df["period"] == "pre"]
        post_df = step_df[step_df["period"] == "post"]
        all_df = step_df

        pre_mean_ic, pre_std_ic, pre_var_ic, pre_count_ic = _agg(pre_df["year_ic"])
        pre_mean_rankic, pre_std_rankic, pre_var_rankic, pre_count_rankic = _agg(pre_df["year_rankic"])
        post_mean_ic, post_std_ic, post_var_ic, post_count_ic = _agg(post_df["year_ic"])
        post_mean_rankic, post_std_rankic, post_var_rankic, post_count_rankic = _agg(post_df["year_rankic"])
        all_mean_ic, all_std_ic, all_var_ic, all_count_ic = _agg(all_df["year_ic"])
        all_mean_rankic, all_std_rankic, all_var_rankic, all_count_rankic = _agg(all_df["year_rankic"])

        generalization_gap = float("nan")
        if not np.isnan(pre_mean_rankic) and not np.isnan(post_mean_rankic):
            generalization_gap = post_mean_rankic - pre_mean_rankic
        # Stability uses standard deviation (not variance): lower std -> higher score.
        stability_score = -post_std_rankic if not np.isnan(post_std_rankic) else float("nan")

        stability_rows.append(
            {
                "run_id": run_id,
                "step": step_pool.step,
                "pool_json": str(step_pool.path),
                "pool_size": len(exprs),
                "pre_year_count": len(pre_df),
                "post_year_count": len(post_df),
                "pre_mean_ic": pre_mean_ic,
                "pre_std_ic": pre_std_ic,
                "pre_var_ic": pre_var_ic,
                "pre_mean_rankic": pre_mean_rankic,
                "pre_std_rankic": pre_std_rankic,
                "pre_var_rankic": pre_var_rankic,
                "post_mean_ic": post_mean_ic,
                "post_std_ic": post_std_ic,
                "post_var_ic": post_var_ic,
                "post_mean_rankic": post_mean_rankic,
                "post_std_rankic": post_std_rankic,
                "post_var_rankic": post_var_rankic,
                # v2 naming aliases (clearer ordering): <period>_<metric>_<agg>
                "pre_ic_mean": pre_mean_ic,
                "pre_ic_std": pre_std_ic,
                "pre_ic_var": pre_var_ic,
                "pre_rankic_mean": pre_mean_rankic,
                "pre_rankic_std": pre_std_rankic,
                "pre_rankic_var": pre_var_rankic,
                "post_ic_mean": post_mean_ic,
                "post_ic_std": post_std_ic,
                "post_ic_var": post_var_ic,
                "post_rankic_mean": post_mean_rankic,
                "post_rankic_std": post_std_rankic,
                "post_rankic_var": post_var_rankic,
                "all_mean_ic": all_mean_ic,
                "all_std_ic": all_std_ic,
                "all_var_ic": all_var_ic,
                "all_mean_rankic": all_mean_rankic,
                "all_std_rankic": all_std_rankic,
                "all_var_rankic": all_var_rankic,
                "all_ic_mean": all_mean_ic,
                "all_ic_std": all_std_ic,
                "all_ic_var": all_var_ic,
                "all_rankic_mean": all_mean_rankic,
                "all_rankic_std": all_std_rankic,
                "all_rankic_var": all_var_rankic,
                "generalization_gap": generalization_gap,
                "stability_score": stability_score,
                "pre_valid_count_ic": pre_count_ic,
                "pre_valid_count_rankic": pre_count_rankic,
                "post_valid_count_ic": post_count_ic,
                "post_valid_count_rankic": post_count_rankic,
                "all_valid_count_ic": all_count_ic,
                "all_valid_count_rankic": all_count_rankic,
                "step_eval_sec": step_elapsed,
                "step_failed_year_count": this_step_failures,
            }
        )

    if not year_rows or not stability_rows:
        raise RuntimeError("evaluation produced empty result")

    year_df = pd.DataFrame(year_rows).sort_values(["step", "year"]).reset_index(drop=True)
    stability_df = pd.DataFrame(stability_rows).sort_values(["step"]).reset_index(drop=True)

    output_dir = run_path / "generalization_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    year_csv = output_dir / "checkpoint_year_metrics.csv"
    stability_csv = output_dir / "checkpoint_stability.csv"
    summary_json = output_dir / "generalization_summary.json"
    manifest_json = output_dir / "eval_manifest.json"
    diff_report_json = output_dir / "diff_report.json"

    year_df.to_csv(year_csv, index=False, encoding="utf-8")
    stability_df.to_csv(stability_csv, index=False, encoding="utf-8")

    if args.consistency_check:
        reference_path = Path(args.consistency_reference).resolve() if args.consistency_reference else summary_json
        if not reference_path.is_file():
            raise RuntimeError(f"consistency reference not found: {reference_path}")
        consistency_reference_payload = _read_json(reference_path)
    else:
        reference_path = Path()

    valid_post = stability_df.dropna(subset=["post_mean_rankic"])
    if not valid_post.empty:
        best_row = valid_post.sort_values("post_mean_rankic", ascending=False).iloc[0]
        best_step = int(best_row["step"])
    else:
        best_row = None
        best_step = None

    gen_summary: Dict[str, object] = {
        "run_id": run_id,
        "run_path": str(run_path),
        "provider_uri": provider_uri,
        "market": args.market,
        "train_start_year": train_start_year,
        "train_end_year": train_end_year,
        "train_window_source": train_window_source,
        "train_window_warnings": window_warnings,
        "pre_years": effective_pre_years,
        "post_years": effective_post_years,
        "evaluated_years": effective_years,
        "checkpoint_count": int(len(stability_df)),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "best_step_by_post_rankic": best_step,
        "best_metrics": (
            {
                "post_mean_rankic": _safe_float(float(best_row["post_mean_rankic"])),
                "post_var_rankic": _safe_float(float(best_row["post_var_rankic"])),
                "post_rankic_mean": _safe_float(float(best_row["post_rankic_mean"])),
                "post_rankic_var": _safe_float(float(best_row["post_rankic_var"])),
                "all_rankic_mean": _safe_float(float(best_row["all_rankic_mean"])),
                "all_rankic_std": _safe_float(float(best_row["all_rankic_std"])),
                "all_ic_mean": _safe_float(float(best_row["all_ic_mean"])),
                "all_ic_std": _safe_float(float(best_row["all_ic_std"])),
                "generalization_gap": _safe_float(float(best_row["generalization_gap"])),
                "stability_score": _safe_float(float(best_row["stability_score"])),
            }
            if best_row is not None
            else {}
        ),
        "artifacts": {
            "checkpoint_year_metrics_csv": str(year_csv),
            "checkpoint_stability_csv": str(stability_csv),
            "eval_manifest_json": str(manifest_json),
        },
    }
    run_duration_sec = max(0.0, time.perf_counter() - run_eval_t0)
    year_build_failures = [
        {
            "year": item.year,
            "status": item.status,
            "error": item.error,
            "elapsed_sec": item.elapsed_sec,
        }
        for item in year_build_stats
        if item.status != "ok"
    ]
    manifest_payload: Dict[str, Any] = {
        "run_id": run_id,
        "run_path": str(run_path),
        "ckpt_dir": str(ckpt_dir),
        "script": str(Path(__file__).resolve()),
        "script_git_revision": _resolve_git_revision(),
        "started_at": run_started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "duration_sec": run_duration_sec,
        "provider_uri": provider_uri,
        "market": args.market,
        "device": str(device),
        "qlib_n_jobs": args.qlib_n_jobs,
        "step_stride": args.step_stride,
        "step_mode": args.step_mode,
        "step_list": _parse_step_list_arg(args.step_list),
        "step_min": args.step_min,
        "step_max": args.step_max,
        "years_arg": args.years,
        "pre_years_limit": args.pre_years_limit,
        "post_years_limit": args.post_years_limit,
        "train_window": {
            "start_year": train_start_year,
            "end_year": train_end_year,
            "source": train_window_source,
            "warnings": window_warnings,
        },
        "evaluable_years_requested": eval_years,
        "evaluable_years_effective": effective_years,
        "effective_pre_years": effective_pre_years,
        "effective_post_years": effective_post_years,
        "checkpoint_total": len(step_pools),
        "checkpoint_evaluated": len(stability_df),
        "timing": {
            "build_year_calculators_sec": calc_build_total_sec,
            "build_year_calculators_cache_hit": calc_build_cache_hit,
            "step_eval_total_sec": float(sum(item["elapsed_sec"] for item in step_timing_rows)),
            "step_eval_rows": step_timing_rows,
            "year_build_rows": [
                {
                    "year": item.year,
                    "elapsed_sec": item.elapsed_sec,
                    "status": item.status,
                    "error": item.error,
                }
                for item in year_build_stats
            ],
        },
        "failures": {
            "year_build_failures": year_build_failures,
            "step_year_failures": step_year_failures,
        },
    }

    summary_metrics_path = run_path / "summary_metrics.json"
    summary_metrics = _read_json(summary_metrics_path)
    if best_row is not None:
        summary_metrics["gen_best_step_by_post_rankic"] = best_step
        summary_metrics["gen_post_rankic_mean_best"] = _safe_float(float(best_row["post_mean_rankic"]))
        summary_metrics["gen_post_rankic_var_best"] = _safe_float(float(best_row["post_var_rankic"]))
        summary_metrics["gen_pre_post_gap_best"] = _safe_float(float(best_row["generalization_gap"]))
        summary_metrics["gen_all_rankic_mean_best"] = _safe_float(float(best_row["all_rankic_mean"]))
        summary_metrics["gen_all_rankic_std_best"] = _safe_float(float(best_row["all_rankic_std"]))
        summary_metrics["gen_all_ic_mean_best"] = _safe_float(float(best_row["all_ic_mean"]))
        summary_metrics["gen_all_ic_std_best"] = _safe_float(float(best_row["all_ic_std"]))
    summary_metrics["gen_train_start_year"] = train_start_year
    summary_metrics["gen_train_end_year"] = train_end_year
    summary_metrics["gen_train_window_source"] = train_window_source
    summary_metrics_path.write_text(
        json.dumps(summary_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    consistency_result: Dict[str, Any] = {
        "enabled": bool(args.consistency_check),
        "reference_path": str(reference_path) if args.consistency_check else "",
        "passed": None,
        "tolerance": args.consistency_tolerance,
        "difference_count": 0,
    }
    if args.consistency_check:
        diff_payload = _compare_consistency(
            reference=consistency_reference_payload,
            current=gen_summary,
            tolerance=max(0.0, args.consistency_tolerance),
        )
        diff_report_json.write_text(json.dumps(diff_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        consistency_result["passed"] = bool(diff_payload["passed"])
        consistency_result["difference_count"] = len(diff_payload.get("differences", []))
        consistency_result["diff_report_path"] = str(diff_report_json)
        if not diff_payload["passed"]:
            print(f"[warn] consistency check failed, see: {diff_report_json}")
    manifest_payload["consistency_check"] = consistency_result

    manifest_json.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_json.write_text(json.dumps(gen_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.consistency_check and consistency_result.get("passed") is False:
        raise RuntimeError(f"consistency check failed for run={run_id}, diff={diff_report_json}")

    return {
        "run_id": run_id,
        "run_path": str(run_path),
        "output_dir": str(output_dir),
        "year_csv": str(year_csv),
        "stability_csv": str(stability_csv),
        "summary_json": str(summary_json),
        "manifest_json": str(manifest_json),
        "summary_metrics_path": str(summary_metrics_path),
        "best_step_by_post_rankic": best_step,
    }


def main() -> None:
    args = _parse_args()
    runs_root = Path(args.runs_root).resolve()
    if not runs_root.is_dir():
        raise FileNotFoundError(f"runs root not found: {runs_root}")

    if args.run_id:
        run_ids = [item.strip() for item in args.run_id.split(",") if item.strip()]
    else:
        run_ids = _list_run_ids(runs_root)
        if not run_ids:
            raise RuntimeError(f"no runs found under: {runs_root}")

    results = []
    failed = []
    calculator_cache: Dict[Tuple[str, str, str, int, int, Tuple[int, ...]], CalculatorCacheEntry] = {}
    for run_id in run_ids:
        try:
            print(f"[start] evaluating run={run_id}")
            result = _evaluate_run(
                run_id=run_id,
                runs_root=runs_root,
                args=args,
                calculator_cache=calculator_cache,
            )
            results.append(result)
            print(f"[done] run={run_id} output={result['output_dir']}")
        except Exception as exc:
            failed.append({"run_id": run_id, "error": str(exc)})
            print(f"[fail] run={run_id}: {exc}")

    payload = {"results": results, "failed": failed}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.run_id and failed:
        raise SystemExit(1)
    if (not args.run_id) and (not results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
