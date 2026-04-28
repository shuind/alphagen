import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

if not hasattr(np, "string_"):
    np.string_ = np.bytes_
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

from tensorboard.backend.event_processing import event_accumulator


RUN_RE = re.compile(
    r"b_seed(?P<seed>\d+)_(?P<backbone>lstm|transformer)_(?P<reward_mode>.+?)_(?P<suffix>\d{14}_\d{14}|\d{14})$"
)

DEFAULT_TAGS = [
    "train/loss",
    "train/policy_gradient_loss",
    "train/value_loss",
    "train/entropy_loss",
    "train/approx_kl",
    "train/clip_fraction",
    "train/explained_variance",
    "rollout/ep_rew_mean",
    "test/rank_ic",
    "time/fps",
]


def parse_run_id(run_id: str) -> Optional[Dict[str, str]]:
    m = RUN_RE.match(run_id)
    if not m:
        return None
    d = m.groupdict()
    d["seed"] = int(d["seed"])
    return d


def latest_event_file(run_dir: Path) -> Optional[Path]:
    files = sorted(run_dir.rglob("events.out.tfevents*"))
    return files[-1] if files else None


def load_scalars(event_path: Path, tags: List[str]) -> List[Dict[str, object]]:
    ea = event_accumulator.EventAccumulator(str(event_path))
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    rows: List[Dict[str, object]] = []
    for tag in tags:
        if tag not in available:
            continue
        for ev in ea.Scalars(tag):
            rows.append(
                {
                    "tag": tag,
                    "step": int(ev.step),
                    "wall_time": float(ev.wall_time),
                    "value": float(ev.value),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tb-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reward-modes", default="re,re+all_v2")
    parser.add_argument("--tags", default=",".join(DEFAULT_TAGS))
    parser.add_argument("--keep-latest-only", action="store_true", default=True)
    args = parser.parse_args()

    tb_root = Path(args.tb_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reward_modes = {x.strip() for x in args.reward_modes.split(",") if x.strip()}
    tags = [x.strip() for x in args.tags.split(",") if x.strip()]

    runs = []
    for run_dir in sorted([p for p in tb_root.iterdir() if p.is_dir()]):
        meta = parse_run_id(run_dir.name)
        if meta is None:
            continue
        if meta["reward_mode"] not in reward_modes:
            continue
        event_path = latest_event_file(run_dir)
        if event_path is None:
            continue
        runs.append(
            {
                "run_id": run_dir.name,
                "run_dir": run_dir,
                "event_path": event_path,
                **meta,
            }
        )

    run_df = pd.DataFrame(runs)
    if run_df.empty:
        raise RuntimeError("no matching runs found")

    if args.keep_latest_only:
        run_df = (
            run_df.sort_values("run_id")
            .groupby(["seed", "backbone", "reward_mode"], as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )

    curve_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for row in run_df.to_dict(orient="records"):
        scalars = load_scalars(Path(row["event_path"]), tags)
        if not scalars:
            continue
        sdf = pd.DataFrame(scalars)
        sdf["run_id"] = row["run_id"]
        sdf["seed"] = row["seed"]
        sdf["backbone"] = row["backbone"]
        sdf["reward_mode"] = row["reward_mode"]
        curve_rows.extend(sdf.to_dict(orient="records"))

        last_df = sdf.sort_values("step").groupby("tag", as_index=False).tail(1)
        summary = {
            "run_id": row["run_id"],
            "seed": row["seed"],
            "backbone": row["backbone"],
            "reward_mode": row["reward_mode"],
        }
        for x in last_df.to_dict(orient="records"):
            key = x["tag"].replace("/", "_")
            summary[key] = x["value"]
            summary[f"{key}_step"] = x["step"]
        summary_rows.append(summary)

    curve_df = pd.DataFrame(curve_rows)
    summary_df = pd.DataFrame(summary_rows)

    curve_csv = output_dir / "tb_compare_curves.csv"
    summary_csv = output_dir / "tb_compare_summary.csv"
    selected_runs_csv = output_dir / "tb_compare_selected_runs.csv"
    meta_json = output_dir / "tb_compare_meta.json"

    curve_df.to_csv(curve_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    run_df.drop(columns=["run_dir", "event_path"]).to_csv(selected_runs_csv, index=False)
    meta_json.write_text(
        json.dumps(
            {
                "tb_root": str(tb_root),
                "reward_modes": sorted(reward_modes),
                "tags": tags,
                "run_count": int(len(run_df)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "curve_csv": str(curve_csv),
                "summary_csv": str(summary_csv),
                "selected_runs_csv": str(selected_runs_csv),
                "meta_json": str(meta_json),
                "run_count": int(len(run_df)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
