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
WINDOW_RE = re.compile(r",(10|20|30|40|50)\)")

STRATEGIES = ("base", "trend", "volatility", "volume", "corr", "rank", "explore")
HEADS = STRATEGIES  # Backward-compatible output aliases for older checkpoints.
RISKY_OPS = {"Div", "Log", "Corr", "Cov", "Std", "Var", "Mad"}
COMPARISON_OPS = {"Greater", "Less"}
TREND_OPS = {"Ref", "Delta", "Mean", "WMA", "EMA", "TSRank"}
VOLATILITY_OPS = {"Std", "Var", "Mad", "SafeSqrt", "Abs", "Max", "Min"}
VOLUME_OPS = {"Mul", "Div", "Mean", "Sum", "Delta"}
CORR_OPS = {"Corr", "Cov"}
RANK_OPS = {"CSRank", "TSRank", "Sign", "Greater", "Less", "Max", "Min"}
VOLUME_FIELDS = {"$volume", "$vwap"}


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
    windows = WINDOW_RE.findall(expr)
    risky = sum(1 for op in ops if op in RISKY_OPS)
    comparisons = sum(1 for op in ops if op in COMPARISON_OPS)
    trend_ops = sum(1 for op in ops if op in TREND_OPS)
    volatility_ops = sum(1 for op in ops if op in VOLATILITY_OPS)
    volume_ops = sum(1 for op in ops if op in VOLUME_OPS)
    corr_ops = sum(1 for op in ops if op in CORR_OPS)
    rank_ops = sum(1 for op in ops if op in RANK_OPS)
    op_count = len(ops)
    unique_fields = set(fields)
    has_range = "$high" in unique_fields and "$low" in unique_fields
    has_volume = bool(unique_fields & VOLUME_FIELDS)
    simple_like = (op_count + len(fields) + len(constants)) <= 8 and _max_depth(expr) <= 4 and (risky / op_count if op_count else 0.0) <= 0.35
    abnormal_nesting = _max_depth(expr) >= 7 and (risky >= 3 or ops.count("Div") >= 3 or ops.count("Log") >= 2)
    return {
        "node_count": float(op_count + len(fields) + len(constants)),
        "depth": float(_max_depth(expr)),
        "char_len": float(len(expr)),
        "op_count": float(op_count),
        "field_count": float(len(fields)),
        "constant_count": float(len(constants)),
        "risky_op_count": float(risky),
        "comparison_op_count": float(comparisons),
        "trend_op_count": float(trend_ops),
        "volatility_op_count": float(volatility_ops),
        "volume_op_count": float(volume_ops),
        "corr_op_count": float(corr_ops),
        "rank_op_count": float(rank_ops),
        "risky_op_ratio": float(risky / op_count) if op_count else 0.0,
        "comparison_op_ratio": float(comparisons / op_count) if op_count else 0.0,
        "trend_op_ratio": float(trend_ops / op_count) if op_count else 0.0,
        "volatility_op_ratio": float(volatility_ops / op_count) if op_count else 0.0,
        "volume_op_ratio": float(volume_ops / op_count) if op_count else 0.0,
        "corr_op_ratio": float(corr_ops / op_count) if op_count else 0.0,
        "rank_op_ratio": float(rank_ops / op_count) if op_count else 0.0,
        "unique_field_count": float(len(unique_fields)),
        "unique_window_count": float(len(set(windows))),
        "has_trend": float(trend_ops > 0),
        "has_volatility": float(volatility_ops > 0 or has_range),
        "has_volume": float(has_volume),
        "has_corr": float(corr_ops > 0),
        "has_rank": float(rank_ops > 0),
        "is_simple_like": float(simple_like),
        "abnormal_nesting": float(abnormal_nesting),
    }


def _strategy_preference_match(strategy: str, stats: Dict[str, float]) -> bool:
    if strategy == "trend":
        return bool(stats["has_trend"])
    if strategy == "volatility":
        return bool(stats["has_volatility"])
    if strategy == "volume":
        return bool(stats["has_volume"])
    if strategy == "corr":
        return bool(stats["has_corr"])
    if strategy == "rank":
        return bool(stats["has_rank"])
    if strategy == "explore":
        return not bool(stats["abnormal_nesting"])
    if strategy == "base":
        return True
    return False


def _head_preference_match(head: str, stats: Dict[str, float]) -> bool:
    return _strategy_preference_match(head, stats)


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
    source_strategies = payload.get("source_strategies", payload.get("source_heads", []))
    source_heads = payload.get("source_heads", source_strategies)
    motif_ids = payload.get("motif_ids", [])
    motif_families = payload.get("motif_families", [])
    edit_paths = payload.get("edit_paths", [])
    naturalness_scores = payload.get("naturalness_scores", [])
    behavior_novelty_scores = payload.get("behavior_novelty_scores", [])
    qd_descriptors = payload.get("qd_descriptors", [])
    robust_scores = payload.get("robust_scores", [])
    type_roots = payload.get("type_roots", [])
    type_styles = payload.get("type_styles", [])
    behavior_clusters = payload.get("behavior_clusters", [])
    meta = _find_run_meta(runs_root, run_id)
    step = _step_from_path(pool_path)

    per_expr = [_expr_stats(expr) for expr in exprs]
    signatures = [_ast_signature(expr) for expr in exprs]
    sig_counts = Counter(signatures)
    op_counts = Counter(op for expr in exprs for op in OP_RE.findall(expr))
    field_counts = Counter(field for expr in exprs for field in FIELD_RE.findall(expr))
    window_counts = Counter(window for expr in exprs for window in WINDOW_RE.findall(expr))
    strategy_counts = Counter(source_strategies[: len(exprs)])
    head_counts = Counter(source_heads[: len(exprs)])
    motif_counts = Counter(str(x) for x in motif_ids[: len(exprs)] if x)
    motif_family_counts = Counter(str(x) for x in motif_families[: len(exprs)] if x)
    qd_counts = Counter(str(x) for x in qd_descriptors[: len(exprs)] if x)
    type_style_counts = Counter(str(x) for x in type_styles[: len(exprs)] if x)
    naturalness_values = [
        float(x) for x in naturalness_scores[: len(exprs)]
        if x is not None and str(x) != ""
    ]
    behavior_values = [
        float(x) for x in behavior_novelty_scores[: len(exprs)]
        if x is not None and str(x) != ""
    ]
    robust_values = [
        float(x) for x in robust_scores[: len(exprs)]
        if x is not None and str(x) != ""
    ]
    aligned = [
        _strategy_preference_match(source_strategies[idx] if idx < len(source_strategies) else "", stats)
        for idx, stats in enumerate(per_expr)
    ]

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
        "mean_unique_field_count": _safe_mean([x["unique_field_count"] for x in per_expr]),
        "mean_unique_window_count": _safe_mean([x["unique_window_count"] for x in per_expr]),
        "risky_op_ratio": _safe_mean([x["risky_op_ratio"] for x in per_expr]),
        "comparison_op_ratio": _safe_mean([x["comparison_op_ratio"] for x in per_expr]),
        "trend_op_ratio": _safe_mean([x["trend_op_ratio"] for x in per_expr]),
        "volatility_op_ratio": _safe_mean([x["volatility_op_ratio"] for x in per_expr]),
        "volume_op_ratio": _safe_mean([x["volume_op_ratio"] for x in per_expr]),
        "corr_op_ratio": _safe_mean([x["corr_op_ratio"] for x in per_expr]),
        "rank_op_ratio": _safe_mean([x["rank_op_ratio"] for x in per_expr]),
        "trend_expr_ratio": _safe_mean([x["has_trend"] for x in per_expr]),
        "volatility_expr_ratio": _safe_mean([x["has_volatility"] for x in per_expr]),
        "volume_expr_ratio": _safe_mean([x["has_volume"] for x in per_expr]),
        "corr_expr_ratio": _safe_mean([x["has_corr"] for x in per_expr]),
        "rank_expr_ratio": _safe_mean([x["has_rank"] for x in per_expr]),
        "simple_expr_ratio": _safe_mean([x["is_simple_like"] for x in per_expr]),
        "classic_strategy_alignment_ratio": (sum(1 for x in aligned if x) / n) if n else 0.0,
        "classic_head_alignment_ratio": (sum(1 for x in aligned if x) / n) if n else 0.0,
        "abnormal_nesting_ratio": _safe_mean([x["abnormal_nesting"] for x in per_expr]),
        "unique_ast_count": len(sig_counts),
        "ast_entropy": _entropy(signatures),
        "ast_max_cluster_ratio": (max(sig_counts.values()) / n) if n else 0.0,
        "top_ops": json.dumps(op_counts.most_common(8), ensure_ascii=False),
        "top_fields": json.dumps(field_counts.most_common(8), ensure_ascii=False),
        "top_windows": json.dumps(window_counts.most_common(8), ensure_ascii=False),
        "source_head_counts": json.dumps(dict(head_counts), ensure_ascii=False),
        "source_strategy_counts": json.dumps(dict(strategy_counts), ensure_ascii=False),
        "motif_counts": json.dumps(dict(motif_counts), ensure_ascii=False),
        "motif_family_counts": json.dumps(dict(motif_family_counts), ensure_ascii=False),
        "top_motifs": json.dumps(motif_counts.most_common(8), ensure_ascii=False),
        "mean_naturalness_score": _safe_mean(naturalness_values),
        "mean_behavior_novelty_score": _safe_mean(behavior_values),
        "motif_trace_ratio": (sum(1 for x in motif_ids[:n] if x) / n) if n else 0.0,
        "qd_coverage": len(qd_counts),
        "qd_descriptor_counts": json.dumps(dict(qd_counts), ensure_ascii=False),
        "top_qd_descriptors": json.dumps(qd_counts.most_common(8), ensure_ascii=False),
        "mean_robust_score": _safe_mean(robust_values),
        "type_style_counts": json.dumps(dict(type_style_counts), ensure_ascii=False),
        "behavior_cluster_count": len(set(x for x in behavior_clusters[:n] if x is not None and str(x) != "")),
    }
    for strategy in STRATEGIES:
        row[f"strategy_{strategy}_count"] = int(strategy_counts.get(strategy, 0))
        row[f"strategy_{strategy}_ratio"] = float(strategy_counts.get(strategy, 0) / n) if n else 0.0
        row[f"head_{strategy}_count"] = int(head_counts.get(strategy, 0))
        row[f"head_{strategy}_ratio"] = float(head_counts.get(strategy, 0) / n) if n else 0.0

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
                "source_strategy": source_strategies[idx] if idx < len(source_strategies) else "",
                "motif_id": motif_ids[idx] if idx < len(motif_ids) else "",
                "motif_family": motif_families[idx] if idx < len(motif_families) else "",
                "edit_path": json.dumps(edit_paths[idx], ensure_ascii=False) if idx < len(edit_paths) else "",
                "naturalness_score": naturalness_scores[idx] if idx < len(naturalness_scores) else "",
                "behavior_novelty_score": behavior_novelty_scores[idx] if idx < len(behavior_novelty_scores) else "",
                "qd_descriptor": qd_descriptors[idx] if idx < len(qd_descriptors) else "",
                "robust_score": robust_scores[idx] if idx < len(robust_scores) else "",
                "type_root": type_roots[idx] if idx < len(type_roots) else "",
                "type_style": type_styles[idx] if idx < len(type_styles) else "",
                "behavior_cluster": behavior_clusters[idx] if idx < len(behavior_clusters) else "",
                "node_count": int(stats["node_count"]),
                "depth": int(stats["depth"]),
                "risky_op_ratio": round(stats["risky_op_ratio"], 4),
                "has_trend": int(stats["has_trend"]),
                "has_volatility": int(stats["has_volatility"]),
                "has_volume": int(stats["has_volume"]),
                "has_corr": int(stats["has_corr"]),
                "has_rank": int(stats["has_rank"]),
                "is_simple_like": int(stats["is_simple_like"]),
                "abnormal_nesting": int(stats["abnormal_nesting"]),
                "classic_head_match": int(_strategy_preference_match(source_strategies[idx] if idx < len(source_strategies) else "", stats)),
                "classic_strategy_match": int(_strategy_preference_match(source_strategies[idx] if idx < len(source_strategies) else "", stats)),
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
            "| run_id | method | step | n | node | depth | risky | trend | volatility | volume | corr | rank | simple | align | abnormal | robust | qd coverage | behavior clusters | natural | novelty | strategy counts | top motifs | top qd |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for row in latest_by_run.values():
        lines.append(
            "| {run_id} | {method} | {step} | {n_factors} | {mean_node_count:.2f} | "
            "{mean_depth:.2f} | {risky_op_ratio:.3f} | {trend_expr_ratio:.3f} | "
            "{volatility_expr_ratio:.3f} | {volume_expr_ratio:.3f} | {corr_expr_ratio:.3f} | "
            "{rank_expr_ratio:.3f} | {simple_expr_ratio:.3f} | "
            "{classic_strategy_alignment_ratio:.3f} | {abnormal_nesting_ratio:.3f} | "
            "{mean_robust_score:.3f} | {qd_coverage} | {behavior_cluster_count} | "
            "{mean_naturalness_score:.3f} | {mean_behavior_novelty_score:.3f} | "
            "`{source_strategy_counts}` | `{top_motifs}` | `{top_qd_descriptors}` |".format(**row)
        )

    lines.extend(["", "## Representative Factors", ""])
    for example in examples:
        if latest_by_run.get(str(example["run_id"]), {}).get("step") != example["step"]:
            continue
        lines.append(
            f"- `{example['run_id']}` step={example['step']} rank={example['rank']} "
            f"strategy={example.get('source_strategy', example['source_head'])} motif={example.get('motif_id', '')} "
            f"qd={example.get('qd_descriptor', '')} robust={example.get('robust_score', '')} "
            f"type={example.get('type_style', '')}/{example.get('type_root', '')} "
            f"family={example.get('motif_family', '')} weight={example['weight']} "
            f"nodes={example['node_count']} depth={example['depth']} risky={example['risky_op_ratio']} "
            f"trend={example['has_trend']} vol={example['has_volatility']} volume={example['has_volume']} "
            f"corr={example['has_corr']} rank={example['has_rank']} simple={example['is_simple_like']} "
            f"align={example['classic_strategy_match']} abnormal={example['abnormal_nesting']} "
            f"natural={example.get('naturalness_score', '')} novelty={example.get('behavior_novelty_score', '')} "
            f"edits={example.get('edit_path', '')}: "
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
