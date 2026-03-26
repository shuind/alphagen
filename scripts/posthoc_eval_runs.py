from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate each checkpoint pool by yearly generalization metrics "
            "(pre-train years vs post-train years)."
        )
    )
    parser.add_argument("--runs-root", required=True, type=str)
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--provider-uri", type=str, default="")
    parser.add_argument("--market", type=str, default="csi300")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
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
    parser.add_argument("--train-start-year", type=int, default=2010)
    parser.add_argument("--train-end-year", type=int, default=2019)
    parser.add_argument("--max-backtrack-days", type=int, default=100)
    parser.add_argument("--max-future-days", type=int, default=30)
    parser.add_argument("--qlib-n-jobs", type=int, default=1)
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


def _list_step_pools(ckpt_dir: Path, step_stride: int) -> List[StepPool]:
    step_pools: List[StepPool] = []
    for path in ckpt_dir.glob("*_steps_pool.json"):
        stem = path.name.split("_steps_pool.json")[0]
        try:
            step = int(stem)
        except ValueError:
            continue
        step_pools.append(StepPool(step=step, path=path))
    step_pools.sort(key=lambda x: x.step)
    if not step_pools:
        return []

    stride = max(1, step_stride)
    if stride == 1:
        return step_pools

    selected = [sp for idx, sp in enumerate(step_pools) if idx % stride == 0]
    if selected[-1].step != step_pools[-1].step:
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
    if cli_provider:
        return cli_provider
    env_provider = Path().absolute()
    _ = env_provider  # keep lint quiet for environments without os import in static analyzers
    import os

    provider = os.environ.get("QLIB_PROVIDER_URI", "")
    if provider:
        return provider
    meta = _read_json(run_path / "run_meta.json")
    if isinstance(meta.get("provider_uri"), str) and meta["provider_uri"]:
        return meta["provider_uri"]
    return "/kaggle/input/baostock/cn_data_baostock_fwdadj"


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
) -> Dict[int, QLibStockDataCalculator]:
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1
    calculators: Dict[int, QLibStockDataCalculator] = {}
    years = list(eval_years)
    total = len(years)
    for idx, year in enumerate(years, start=1):
        start_time = f"{year}-01-01"
        end_time = f"{year}-12-31"
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
                continue
            calculators[year] = calculator
        except Exception as exc:
            print(f"[warn] skip year={year}: {exc}")
    return calculators


def _evaluate_run(
    run_id: str,
    runs_root: Path,
    args: argparse.Namespace,
) -> Dict:
    run_path = _resolve_run_path(runs_root, run_id)
    if run_path is None:
        raise FileNotFoundError(f"run not found: {run_id}")
    ckpt_dir = _resolve_ckpt_dir(runs_root, run_path, run_id)
    if ckpt_dir is None:
        raise FileNotFoundError(f"checkpoint dir not found for run: {run_id}")

    provider_uri = _resolve_provider_uri(args.provider_uri, run_path)
    _init_qlib(provider_uri, args.qlib_n_jobs)
    device = _resolve_device(args.device)

    pre_years, post_years, eval_years = _collect_eval_years(
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
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

    calculators = _build_year_calculators(
        market=args.market,
        eval_years=eval_years,
        device=device,
        max_backtrack_days=args.max_backtrack_days,
        max_future_days=args.max_future_days,
    )
    if not calculators:
        raise RuntimeError("no yearly calculators built successfully")
    effective_years = sorted(list(calculators.keys()))
    effective_pre_years = [y for y in effective_years if y < args.train_start_year]
    effective_post_years = [y for y in effective_years if y > args.train_end_year]

    step_pools = _list_step_pools(ckpt_dir, args.step_stride)
    if not step_pools:
        raise FileNotFoundError(f"no *_steps_pool.json under {ckpt_dir}")

    year_rows: List[Dict] = []
    stability_rows: List[Dict] = []

    for idx, step_pool in enumerate(step_pools, start=1):
        print(f"[step] evaluating checkpoint {idx}/{len(step_pools)}: step={step_pool.step}")
        exprs, weights = load_alpha_pool_by_path(str(step_pool.path))
        if len(exprs) == 0:
            continue
        if len(exprs) != len(weights):
            m = min(len(exprs), len(weights))
            exprs, weights = exprs[:m], weights[:m]

        this_step_rows: List[Dict] = []
        for year, calculator in calculators.items():
            metrics = _eval_pool_on_calculator(calculator, exprs, weights)
            period = "pre" if year < args.train_start_year else "post"
            row = {
                "run_id": run_id,
                "step": step_pool.step,
                "pool_json": str(step_pool.path),
                "pool_size": len(exprs),
                "year": year,
                "period": period,
                **metrics,
            }
            year_rows.append(row)
            this_step_rows.append(row)

        step_df = pd.DataFrame(this_step_rows)
        pre_df = step_df[step_df["period"] == "pre"]
        post_df = step_df[step_df["period"] == "post"]

        pre_mean_ic, pre_std_ic, pre_var_ic, pre_count_ic = _agg(pre_df["year_ic"])
        pre_mean_rankic, pre_std_rankic, pre_var_rankic, pre_count_rankic = _agg(pre_df["year_rankic"])
        post_mean_ic, post_std_ic, post_var_ic, post_count_ic = _agg(post_df["year_ic"])
        post_mean_rankic, post_std_rankic, post_var_rankic, post_count_rankic = _agg(post_df["year_rankic"])

        generalization_gap = float("nan")
        if not np.isnan(pre_mean_rankic) and not np.isnan(post_mean_rankic):
            generalization_gap = post_mean_rankic - pre_mean_rankic
        stability_score = -post_var_rankic if not np.isnan(post_var_rankic) else float("nan")

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
                "generalization_gap": generalization_gap,
                "stability_score": stability_score,
                "pre_valid_count_ic": pre_count_ic,
                "pre_valid_count_rankic": pre_count_rankic,
                "post_valid_count_ic": post_count_ic,
                "post_valid_count_rankic": post_count_rankic,
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

    year_df.to_csv(year_csv, index=False, encoding="utf-8")
    stability_df.to_csv(stability_csv, index=False, encoding="utf-8")

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
        "train_start_year": args.train_start_year,
        "train_end_year": args.train_end_year,
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
                "generalization_gap": _safe_float(float(best_row["generalization_gap"])),
                "stability_score": _safe_float(float(best_row["stability_score"])),
            }
            if best_row is not None
            else {}
        ),
        "artifacts": {
            "checkpoint_year_metrics_csv": str(year_csv),
            "checkpoint_stability_csv": str(stability_csv),
        },
    }
    summary_json.write_text(json.dumps(gen_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_metrics_path = run_path / "summary_metrics.json"
    summary_metrics = _read_json(summary_metrics_path)
    if best_row is not None:
        summary_metrics["gen_best_step_by_post_rankic"] = best_step
        summary_metrics["gen_post_rankic_mean_best"] = _safe_float(float(best_row["post_mean_rankic"]))
        summary_metrics["gen_post_rankic_var_best"] = _safe_float(float(best_row["post_var_rankic"]))
        summary_metrics["gen_pre_post_gap_best"] = _safe_float(float(best_row["generalization_gap"]))
    summary_metrics_path.write_text(
        json.dumps(summary_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "run_id": run_id,
        "run_path": str(run_path),
        "output_dir": str(output_dir),
        "year_csv": str(year_csv),
        "stability_csv": str(stability_csv),
        "summary_json": str(summary_json),
        "summary_metrics_path": str(summary_metrics_path),
        "best_step_by_post_rankic": best_step,
    }


def main() -> None:
    args = _parse_args()
    runs_root = Path(args.runs_root).resolve()
    if not runs_root.is_dir():
        raise FileNotFoundError(f"runs root not found: {runs_root}")

    if args.run_id:
        run_ids = [args.run_id]
    else:
        run_ids = _list_run_ids(runs_root)
        if not run_ids:
            raise RuntimeError(f"no runs found under: {runs_root}")

    results = []
    failed = []
    for run_id in run_ids:
        try:
            print(f"[start] evaluating run={run_id}")
            result = _evaluate_run(run_id=run_id, runs_root=runs_root, args=args)
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
