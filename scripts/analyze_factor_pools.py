import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


OP_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]*)\(")
FIELD_RE = re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*")
CONSTANT_RE = re.compile(r"Constant\([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?\)", re.IGNORECASE)
STEP_RE = re.compile(r"(\d+)_steps_pool\.json$")

HEADS = ("base", "simple", "ts", "pv", "rank", "explore")
RISKY_OPS = {"Div", "Log", "Corr", "Cov", "Std", "Var", "Mad"}
COMPARISON_OPS = {"Greater", "Less"}
TS_OPS = {"Ref", "Delta", "Mean", "Std", "Corr", "Cov", "WMA", "EMA", "Sum", "Med"}


def _split_csv(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _step_from_path(path: Path) -> int:
    match = STEP_RE.search(path.name)
    if not match:
        return -1
    return int(match.group(1))


def _resolve_checkpoint_root(runs_root: Path) -> Path:
    if (runs_root / "checkpoints").exists():
        return runs_root / "checkpoints"
    return runs_root


def _find_run_meta(runs_root: Path, run_id: str) -> Dict:
    for candidate in (runs_root / "runs" / run_id / "run_meta.json", runs_root / run_id / "run_meta.json"):
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                return {}
    return {}


def _iter_pool_files(checkpoint_root: Path, run_ids: Optional[Iterable[str]]) -> Iterable[Tuple[str, Path]]:
    wanted = set(run_ids or [])
    for run_dir in sorted(path for path in checkpoint_root.iterdir() if path.is_dir()):
        if wanted and run_dir.name not in wanted:
            continue
        files = sorted(run_dir.glob("*_steps_pool.json"), key=_step_from_path)
        for path in files:
            yield run_dir.name, path


def _max_depth(expr: str) -> int:
    depth = 0
    best = 0
    for ch in expr:
        if ch == "(":
            depth += 1
            best = max(best, depth)
        elif ch == ")":
            depth = max(0, depth - 1)
    return best


def _ast_signature(expr: str) -> str:
    text = CONSTANT_RE.sub("CONST", expr)
    text = FIELD_RE.sub("FIELD", text)
    text = re.sub(r"\s+", "", text)
    return text


def _entropy(values: Iterable[str]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def _safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _expr_stats(expr: str) -> Dict[str, float]:
    ops = OP_RE.findall(expr)
    fields = FIELD_RE.findall(expr)
    constants = CONSTANT_RE.findall(expr)
    risky = sum(1 for op in ops if op in RISKY_OPS)
    comparisons = sum(1 for op in ops if op in COMPARISON_OPS)
    ts_ops = sum(1 for op in ops if op in TS_OPS)
    op_count = len(ops)
    return {
        "node_count": float(op_count + len(fields) + len(constants)),
        "depth": float(_max_depth(expr)),
        "char_len": float(len(expr)),
        "op_count": float(op_count),
        "field_count": float(len(fields)),
        "constant_count": float(len(constants)),
        "risky_op_count": float(risky),
        "comparison_op_count": float(comparisons),
        "ts_op_count": float(ts_ops),
        "risky_op_ratio": float(risky / op_count) if op_count else 0.0,
        "comparison_op_ratio": float(comparisons / op_count) if op_count else 0.0,
        "ts_op_ratio": float(ts_ops / op_count) if op_count else 0.0,
    }


def _select_files(files: List[Tuple[str, Path]], step_mode: str) -> List[Tuple[str, Path]]:
    if step_mode == "all":
        return files
    latest: Dict[str, Tuple[str, Path]] = {}
    for run_id, path in files:
        old = latest.get(run_id)
        if old is None or _step_from_path(path) > _step_from_path(old[1]):
            latest[run_id] = (run_id, path)
    return [latest[key] for key in sorted(latest)]


def _summarize_pool(runs_root: Path, run_id: str, pool_path: Path) -> Tuple[Dict, List[Dict]]:
    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    exprs = [str(x) for x in payload.get("exprs", []) if str(x)]
    weights = payload.get("weights", [])
    source_heads = payload.get("source_heads", [])
    meta = _find_run_meta(runs_root, run_id)
    step = _step_from_path(pool_path)

    per_expr = [_expr_stats(expr) for expr in exprs]
    signatures = [_ast_signature(expr) for expr in exprs]
    sig_counts = Counter(signatures)
    op_counts = Counter(op for expr in exprs for op in OP_RE.findall(expr))
    field_counts = Counter(field for expr in exprs for field in FIELD_RE.findall(expr))
    head_counts = Counter(source_heads[: len(exprs)])

    n = len(exprs)
    row = {
        "run_id": run_id,
        "step": step,
        "method": meta.get("method", ""),
        "backbone": meta.get("backbone", ""),
        "seed": meta.get("seed", ""),
        "pool_capacity": meta.get("pool_capacity", n),
        "n_factors": n,
        "mean_node_count": _safe_mean([x["node_count"] for x in per_expr]),
        "mean_depth": _safe_mean([x["depth"] for x in per_expr]),
        "mean_char_len": _safe_mean([x["char_len"] for x in per_expr]),
        "mean_op_count": _safe_mean([x["op_count"] for x in per_expr]),
        "mean_field_count": _safe_mean([x["field_count"] for x in per_expr]),
        "mean_constant_count": _safe_mean([x["constant_count"] for x in per_expr]),
        "risky_op_ratio": _safe_mean([x["risky_op_ratio"] for x in per_expr]),
        "comparison_op_ratio": _safe_mean([x["comparison_op_ratio"] for x in per_expr]),
        "ts_op_ratio": _safe_mean([x["ts_op_ratio"] for x in per_expr]),
        "unique_ast_count": len(sig_counts),
        "ast_entropy": _entropy(signatures),
        "ast_max_cluster_ratio": (max(sig_counts.values()) / n) if n else 0.0,
        "top_ops": json.dumps(op_counts.most_common(8), ensure_ascii=False),
        "top_fields": json.dumps(field_counts.most_common(8), ensure_ascii=False),
        "source_head_counts": json.dumps(dict(head_counts), ensure_ascii=False),
    }
    for head in HEADS:
        row[f"head_{head}_count"] = int(head_counts.get(head, 0))
        row[f"head_{head}_ratio"] = float(head_counts.get(head, 0) / n) if n else 0.0

    examples: List[Dict] = []
    order = list(range(n))
    if len(weights) >= n:
        order.sort(key=lambda idx: abs(float(weights[idx])), reverse=True)
    for idx in order[: min(8, n)]:
        stats = per_expr[idx]
        examples.append(
            {
                "run_id": run_id,
                "step": step,
                "rank": len(examples) + 1,
                "weight": float(weights[idx]) if idx < len(weights) else "",
                "source_head": source_heads[idx] if idx < len(source_heads) else "",
                "node_count": int(stats["node_count"]),
                "depth": int(stats["depth"]),
                "risky_op_ratio": round(stats["risky_op_ratio"], 4),
                "expr": exprs[idx],
            }
        )
    return row, examples


def _write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: List[Dict], examples: List[Dict]) -> None:
    lines = ["# Factor Pool Direct Analysis", ""]
    if not rows:
        lines.append("No checkpoint pool files found.")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    latest_by_run: Dict[str, Dict] = {}
    for row in rows:
        old = latest_by_run.get(row["run_id"])
        if old is None or int(row["step"]) > int(old["step"]):
            latest_by_run[str(row["run_id"])] = row

    lines.extend(
        [
            "| run_id | method | step | n | node | depth | risky | compare | ts | AST entropy | head counts |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in latest_by_run.values():
        lines.append(
            "| {run_id} | {method} | {step} | {n_factors} | {mean_node_count:.2f} | "
            "{mean_depth:.2f} | {risky_op_ratio:.3f} | {comparison_op_ratio:.3f} | "
            "{ts_op_ratio:.3f} | {ast_entropy:.3f} | `{source_head_counts}` |".format(**row)
        )

    lines.extend(["", "## Representative Factors", ""])
    for example in examples:
        if latest_by_run.get(str(example["run_id"]), {}).get("step") != example["step"]:
            continue
        lines.append(
            f"- `{example['run_id']}` step={example['step']} rank={example['rank']} "
            f"head={example['source_head']} weight={example['weight']} "
            f"nodes={example['node_count']} depth={example['depth']} risky={example['risky_op_ratio']}: "
            f"`{example['expr']}`"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze AlphaPool checkpoint factor expressions directly.")
    parser.add_argument("--runs-root", default="data", help="root containing checkpoints/ and optionally runs/")
    parser.add_argument("--run-id", default="", help="comma-separated run ids; default all runs under checkpoints")
    parser.add_argument("--step-mode", choices=["latest", "all"], default="latest")
    parser.add_argument("--output-dir", default="data/factor_pool_analysis")
    args = parser.parse_args()

    runs_root = Path(args.runs_root).resolve()
    checkpoint_root = _resolve_checkpoint_root(runs_root)
    run_ids = _split_csv(args.run_id) if args.run_id else None
    files = _select_files(list(_iter_pool_files(checkpoint_root, run_ids)), args.step_mode)

    rows: List[Dict] = []
    examples: List[Dict] = []
    for run_id, path in files:
        row, pool_examples = _summarize_pool(runs_root, run_id, path)
        rows.append(row)
        examples.extend(pool_examples)

    output_dir = Path(args.output_dir).resolve()
    _write_csv(output_dir / "factor_pool_summary.csv", rows)
    _write_csv(output_dir / "factor_pool_examples.csv", examples)
    _write_markdown(output_dir / "factor_pool_report.md", rows, examples)

    print(f"[done] pools={len(rows)} examples={len(examples)} output={output_dir}")


if __name__ == "__main__":
    main()
