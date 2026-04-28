from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate and visualize generalization eval results.")
    parser.add_argument("--runs-root", type=str, required=True, help="Path containing run folders.")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save csv/png outputs.")
    return parser.parse_args()


def _read_json(path: Path) -> Dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_run_id(run_id: str) -> Dict[str, str]:
    # Example: b_seed0_lstm_re+func_v2_20260331130957_20260331051219
    m = re.match(r"^b_seed(?P<seed>\d+)_(?P<backbone>lstm|transformer)_(?P<reward>.+)_\d{14}_\d{14}$", run_id)
    if not m:
        return {"seed": "", "backbone": "", "reward_mode": ""}
    return {
        "seed": m.group("seed"),
        "backbone": m.group("backbone"),
        "reward_mode": m.group("reward"),
    }


def _collect_rows(runs_root: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for run_dir in sorted([p for p in runs_root.iterdir() if p.is_dir()]):
        run_id = run_dir.name
        summary_path = run_dir / "generalization_eval" / "generalization_summary.json"
        summary = _read_json(summary_path)
        best_metrics = summary.get("best_metrics", {}) if isinstance(summary.get("best_metrics", {}), dict) else {}
        meta = _parse_run_id(run_id)
        rows.append(
            {
                "run_id": run_id,
                "seed": meta["seed"],
                "backbone": meta["backbone"],
                "reward_mode": meta["reward_mode"],
                "best_step_by_post_rankic": summary.get("best_step_by_post_rankic"),
                "post_rankic_mean": best_metrics.get("post_rankic_mean"),
                "post_rankic_var": best_metrics.get("post_rankic_var"),
                "all_rankic_mean": best_metrics.get("all_rankic_mean"),
                "all_rankic_std": best_metrics.get("all_rankic_std"),
                "all_ic_mean": best_metrics.get("all_ic_mean"),
                "all_ic_std": best_metrics.get("all_ic_std"),
                "generalization_gap": best_metrics.get("generalization_gap"),
                "stability_score": best_metrics.get("stability_score"),
                "summary_path": str(summary_path),
            }
        )
    df = pd.DataFrame(rows)
    numeric_cols = [
        "best_step_by_post_rankic",
        "post_rankic_mean",
        "post_rankic_var",
        "all_rankic_mean",
        "all_rankic_std",
        "all_ic_mean",
        "all_ic_std",
        "generalization_gap",
        "stability_score",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _plot_run_rankic(df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = df.sort_values("post_rankic_mean", ascending=False).reset_index(drop=True)
    fig_h = max(6, 0.35 * len(plot_df))
    plt.figure(figsize=(12, fig_h))
    plt.barh(plot_df["run_id"], plot_df["post_rankic_mean"])
    plt.gca().invert_yaxis()
    plt.xlabel("post_rankic_mean")
    plt.title("Run-level Post RankIC Mean")
    plt.tight_layout()
    plt.savefig(output_dir / "run_post_rankic_mean.png", dpi=150)
    plt.close()


def _plot_mode_summary(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    agg = (
        df.groupby("reward_mode", dropna=False)["post_rankic_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "post_rankic_mean_mean", "std": "post_rankic_mean_std"})
        .sort_values("post_rankic_mean_mean", ascending=False)
    )
    plt.figure(figsize=(10, 5))
    plt.bar(agg["reward_mode"], agg["post_rankic_mean_mean"], yerr=agg["post_rankic_mean_std"].fillna(0.0), capsize=4)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("post_rankic_mean")
    plt.title("Reward Mode Comparison (mean ± std)")
    plt.tight_layout()
    plt.savefig(output_dir / "reward_mode_post_rankic.png", dpi=150)
    plt.close()
    return agg


def _plot_backbone_summary(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    agg = (
        df.groupby("backbone", dropna=False)["post_rankic_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "post_rankic_mean_mean", "std": "post_rankic_mean_std"})
        .sort_values("post_rankic_mean_mean", ascending=False)
    )
    plt.figure(figsize=(6, 4))
    plt.bar(agg["backbone"], agg["post_rankic_mean_mean"], yerr=agg["post_rankic_mean_std"].fillna(0.0), capsize=4)
    plt.ylabel("post_rankic_mean")
    plt.title("Backbone Comparison (mean ± std)")
    plt.tight_layout()
    plt.savefig(output_dir / "backbone_post_rankic.png", dpi=150)
    plt.close()
    return agg


def _plot_gap_vs_stability(df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = df.dropna(subset=["generalization_gap", "stability_score", "post_rankic_mean"]).copy()
    plt.figure(figsize=(7, 5))
    plt.scatter(plot_df["generalization_gap"], plot_df["stability_score"], s=40)
    for _, row in plot_df.iterrows():
        label = f"{row['backbone']}/{row['reward_mode']}/s{row['seed']}"
        plt.annotate(label, (row["generalization_gap"], row["stability_score"]), fontsize=7, alpha=0.8)
    plt.xlabel("generalization_gap (post - pre)")
    plt.ylabel("stability_score (-post_rankic_std)")
    plt.title("Gap vs Stability")
    plt.tight_layout()
    plt.savefig(output_dir / "gap_vs_stability.png", dpi=150)
    plt.close()


def main() -> None:
    args = _parse_args()
    runs_root = Path(args.runs_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _collect_rows(runs_root)
    if df.empty:
        raise RuntimeError(f"no run folders found under: {runs_root}")

    run_csv = output_dir / "generalization_overview_runs.csv"
    df.to_csv(run_csv, index=False, encoding="utf-8")

    mode_df = _plot_mode_summary(df, output_dir)
    mode_df.to_csv(output_dir / "generalization_overview_by_reward_mode.csv", index=False, encoding="utf-8")

    backbone_df = _plot_backbone_summary(df, output_dir)
    backbone_df.to_csv(output_dir / "generalization_overview_by_backbone.csv", index=False, encoding="utf-8")

    _plot_run_rankic(df, output_dir)
    _plot_gap_vs_stability(df, output_dir)

    payload = {
        "runs_root": str(runs_root),
        "output_dir": str(output_dir),
        "run_count": int(len(df)),
        "run_csv": str(run_csv),
        "figures": [
            str(output_dir / "run_post_rankic_mean.png"),
            str(output_dir / "reward_mode_post_rankic.png"),
            str(output_dir / "backbone_post_rankic.png"),
            str(output_dir / "gap_vs_stability.png"),
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
