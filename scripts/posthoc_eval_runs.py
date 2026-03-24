from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
LOCAL_UI_ROOT = PROJECT_ROOT / "local_ui"
if str(LOCAL_UI_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_UI_ROOT))

from lib.io_runs import RunInfo, scan_runs  # noqa: E402

from alphagen.data.expression import Feature, Ref  # noqa: E402
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr  # noqa: E402
from alphagen_qlib.calculator import QLibStockDataCalculator  # noqa: E402
from alphagen_qlib.compat import patch_all  # noqa: E402
from alphagen_qlib.stock_data import FeatureType, StockData  # noqa: E402
from alphagen_qlib.utils import load_alpha_pool_by_path  # noqa: E402


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def _resolve_provider_uri(cli_value: str) -> str:
    return (
        cli_value
        or os.environ.get("QLIB_PROVIDER_URI", "")
        or os.path.expanduser("~/.qlib/qlib_data/cn_data_baostock_fwdadj")
    )


def _init_qlib(provider_uri: str) -> None:
    try:
        patch_all(provider_uri)
        import qlib
        from qlib.constant import REG_CN
    except ImportError as exc:  # pragma: no cover - operational path
        raise RuntimeError(
            "posthoc evaluation requires qlib/pyqlib in the current Python environment"
        ) from exc

    qlib.init(provider_uri=provider_uri, region=REG_CN)


def _latest_pool_json(ckpt_dir: str) -> Tuple[Optional[Path], Optional[int]]:
    path = Path(ckpt_dir)
    if not path.is_dir():
        return None, None
    candidates = []
    for p in path.glob("*_steps_pool.json"):
        try:
            step = int(p.name.split("_steps_pool.json")[0])
        except ValueError:
            continue
        candidates.append((step, p))
    if not candidates:
        return None, None
    step, best_path = max(candidates, key=lambda x: x[0])
    return best_path, step


def _build_target():
    close = Feature(FeatureType.CLOSE)
    return Ref(close, -20) / close - 1


def _calc_ic_family(
    calculator: QLibStockDataCalculator,
    exprs,
    weights,
) -> Dict[str, Optional[float]]:
    ensemble = calculator.make_ensemble_alpha(exprs, weights)
    target = calculator.target_value
    if target is None:
        return {
            "test_ic": None,
            "test_rankic": None,
            "test_icir": None,
            "test_rankicir": None,
        }

    daily_ic = batch_pearsonr(ensemble, target).detach().cpu().numpy()
    daily_rankic = batch_spearmanr(ensemble, target).detach().cpu().numpy()

    def _ir(values) -> Optional[float]:
        std = float(values.std())
        if std < 1e-12:
            return None
        return float(values.mean() / std)

    return {
        "test_ic": _safe_float(float(daily_ic.mean())),
        "test_rankic": _safe_float(float(daily_rankic.mean())),
        "test_icir": _safe_float(_ir(daily_ic)),
        "test_rankicir": _safe_float(_ir(daily_rankic)),
    }


def _calc_diversity(
    calculator: QLibStockDataCalculator,
    exprs,
    weights: List[float],
) -> Dict[str, Optional[float]]:
    significant = [(expr, float(w)) for expr, w in zip(exprs, weights) if abs(float(w)) > 1e-4]
    active_exprs = [expr for expr, _ in significant]
    active_weights = [w for _, w in significant]

    if len(active_exprs) < 2:
        return {
            "factor_count": len(exprs),
            "significant_factor_count": len(active_exprs),
            "avg_abs_mutual_ic": 0.0,
            "max_abs_mutual_ic": 0.0,
            "diversity_score": 1.0,
            "weight_l1": _safe_float(sum(abs(w) for w in active_weights)),
        }

    pair_abs_ics: List[float] = []
    for i in range(len(active_exprs)):
        for j in range(i + 1, len(active_exprs)):
            pair_abs_ics.append(abs(float(calculator.calc_mutual_IC(active_exprs[i], active_exprs[j]))))

    avg_abs_mutual_ic = float(sum(pair_abs_ics) / len(pair_abs_ics)) if pair_abs_ics else 0.0
    max_abs_mutual_ic = float(max(pair_abs_ics)) if pair_abs_ics else 0.0
    return {
        "factor_count": len(exprs),
        "significant_factor_count": len(active_exprs),
        "avg_abs_mutual_ic": _safe_float(avg_abs_mutual_ic),
        "max_abs_mutual_ic": _safe_float(max_abs_mutual_ic),
        "diversity_score": _safe_float(1.0 - avg_abs_mutual_ic),
        "weight_l1": _safe_float(sum(abs(w) for w in active_weights)),
    }


def _calc_backtest(
    calculator: QLibStockDataCalculator,
    data: StockData,
    exprs,
    weights,
) -> Dict[str, object]:
    from backtest import QlibBacktest

    ensemble = calculator.make_ensemble_alpha(exprs, weights)
    signal_df = data.make_dataframe(ensemble)
    signal = signal_df["0"]
    backtester = QlibBacktest()
    try:
        result = backtester.run(signal, return_report=False)
    except Exception as exc:  # pragma: no cover - operational path
        return {"backtest_error": str(exc)}
    return {
        "sharpe": _safe_float(result.get("sharpe")),
        "annual_return": _safe_float(result.get("annual_return")),
        "max_drawdown": _safe_float(result.get("max_drawdown")),
        "information_ratio": _safe_float(result.get("information_ratio")),
        "annual_excess_return": _safe_float(result.get("annual_excess_return")),
        "excess_max_drawdown": _safe_float(result.get("excess_max_drawdown")),
    }


def _evaluate_run(run: RunInfo, provider_uri: str) -> Dict[str, object]:
    pool_path, pool_step = _latest_pool_json(run.ckpt_dir)
    if pool_path is None:
        raise FileNotFoundError(f"no *_steps_pool.json found under {run.ckpt_dir}")

    meta = run.meta or {}
    market = str(meta.get("market") or "csi300")
    target = _build_target()

    exprs, weights = load_alpha_pool_by_path(str(pool_path))
    test_data = StockData(
        instrument=market,
        start_time="2021-01-01",
        end_time="2022-12-31",
        device=torch.device("cpu"),
    )
    test_calculator = QLibStockDataCalculator(test_data, target)

    summary: Dict[str, object] = {
        "run_id": run.run_id,
        "market": market,
        "pool_json": str(pool_path),
        "pool_step": pool_step,
        "provider_uri": provider_uri,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    summary.update(_calc_ic_family(test_calculator, exprs, weights))
    summary.update(_calc_diversity(test_calculator, exprs, list(weights)))
    summary.update(_calc_backtest(test_calculator, test_data, exprs, weights))
    return summary


def _write_summary(run: RunInfo, summary: Dict[str, object]) -> Path:
    output = Path(run.path) / "summary_metrics.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default=str(PROJECT_ROOT / "kaggle" / "working"))
    parser.add_argument("--provider-uri", default="")
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    provider_uri = _resolve_provider_uri(args.provider_uri)
    os.environ["QLIB_PROVIDER_URI"] = provider_uri
    _init_qlib(provider_uri)

    runs = scan_runs(args.runs_root)
    if args.run_id:
        runs = [run for run in runs if run.run_id == args.run_id]
    if not runs:
        raise SystemExit("no runs found")

    for run in runs:
        print(f"[posthoc] evaluating {run.run_id}")
        summary = _evaluate_run(run, provider_uri)
        output = _write_summary(run, summary)
        print(f"[posthoc] wrote {output}")


if __name__ == "__main__":
    main()
