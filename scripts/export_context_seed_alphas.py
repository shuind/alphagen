from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphagen_embedding.dataset import collect_alpha_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a compact alpha seed file for contextual training.")
    parser.add_argument("--checkpoint_root", type=str, default="kaggle/working/checkpoints")
    parser.add_argument("--runs_root", type=str, default="kaggle/working/runs")
    parser.add_argument("--output", type=str, default="context_seed_alphas.json")
    parser.add_argument("--num_alpha", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = collect_alpha_candidates(
        checkpoint_root=args.checkpoint_root,
        runs_root=args.runs_root,
        min_alpha=args.num_alpha,
        max_alpha=args.num_alpha,
    )
    payload = {
        "meta": {
            "source": "checkpoint_scan",
            "num_alpha": len(candidates),
        },
        "alphas": [
            {
                "expr_text": row["expr_text"],
                "run_id": row["run_id"],
                "pool_path": row["pool_path"],
                "weight": float(row["weight"]),
                "test_rankic": float(row["test_rankic"]),
                "diversity_score": float(row["diversity_score"]),
                "order": int(row["order"]),
            }
            for row in candidates
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps({"output": str(output), "num_alpha": len(candidates)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
