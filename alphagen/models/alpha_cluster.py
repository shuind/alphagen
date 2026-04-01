from __future__ import annotations

import math
from collections import Counter, OrderedDict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from alphagen.data.expression import (
    BinaryOperator,
    Constant,
    Corr,
    Cov,
    Div,
    Expression,
    Feature,
    Greater,
    Less,
    PairRollingOperator,
    RollingOperator,
    UnaryOperator,
)


AST_FEATURE_FIELDS: Tuple[str, ...] = ("open", "high", "low", "close", "volume", "vwap")
AST_OPERATOR_NAMES: Tuple[str, ...] = tuple(
    sorted(
        [
            "Abs",
            "Add",
            "Corr",
            "Cov",
            "CSRank",
            "Delta",
            "Div",
            "EMA",
            "Greater",
            "Kurt",
            "Less",
            "Log",
            "Mad",
            "Max",
            "Mean",
            "Med",
            "Min",
            "Mul",
            "Pow",
            "Rank",
            "Ref",
            "Sign",
            "Skew",
            "Std",
            "Sub",
            "Sum",
            "Var",
            "WMA",
        ]
    )
)
RISKY_OPERATOR_NAMES = {"Div", "Log", "Less", "Greater", "Corr", "Cov"}


def iter_expression_nodes(expr: Expression) -> Iterable[Expression]:
    yield expr
    if isinstance(expr, UnaryOperator):
        yield from iter_expression_nodes(expr._operand)
    elif isinstance(expr, BinaryOperator):
        yield from iter_expression_nodes(expr._lhs)
        yield from iter_expression_nodes(expr._rhs)
    elif isinstance(expr, RollingOperator):
        yield from iter_expression_nodes(expr._operand)
    elif isinstance(expr, PairRollingOperator):
        yield from iter_expression_nodes(expr._lhs)
        yield from iter_expression_nodes(expr._rhs)


def expression_depth(expr: Expression) -> int:
    if isinstance(expr, UnaryOperator):
        return 1 + expression_depth(expr._operand)
    if isinstance(expr, BinaryOperator):
        return 1 + max(expression_depth(expr._lhs), expression_depth(expr._rhs))
    if isinstance(expr, RollingOperator):
        return 1 + expression_depth(expr._operand)
    if isinstance(expr, PairRollingOperator):
        return 1 + max(expression_depth(expr._lhs), expression_depth(expr._rhs))
    return 1


def ast_feature_names() -> List[str]:
    names = [
        "node_count",
        "leaf_count",
        "max_depth",
        "avg_branching_factor",
        "window_count",
        "window_min",
        "window_max",
        "window_mean",
        "window_std",
        "risky_op_count",
        "div_count",
        "log_count",
        "less_greater_count",
        "corr_cov_count",
        "feature_ratio",
        "constant_ratio",
        "unary_ratio",
        "binary_ratio",
        "rolling_ratio",
        "pair_rolling_ratio",
    ]
    names.extend([f"field_ratio_{name}" for name in AST_FEATURE_FIELDS])
    names.extend([f"op_ratio_{name.lower()}" for name in AST_OPERATOR_NAMES])
    return names


def extract_ast_feature_map(expr: Expression) -> "OrderedDict[str, float]":
    nodes = list(iter_expression_nodes(expr))
    node_count = len(nodes)
    leaf_count = 0
    feature_count = 0
    constant_count = 0
    unary_count = 0
    binary_count = 0
    rolling_count = 0
    pair_rolling_count = 0
    total_children = 0
    window_values: List[float] = []
    risky_op_count = 0
    div_count = 0
    log_count = 0
    less_greater_count = 0
    corr_cov_count = 0
    field_counter: Counter[str] = Counter()
    op_counter: Counter[str] = Counter()

    for node in nodes:
        if isinstance(node, Feature):
            feature_count += 1
            field_name = node._feature.name.lower()
            if field_name in AST_FEATURE_FIELDS:
                field_counter[field_name] += 1
            leaf_count += 1
        elif isinstance(node, Constant):
            constant_count += 1
            leaf_count += 1

        if isinstance(node, UnaryOperator):
            unary_count += 1
            total_children += 1
        elif isinstance(node, BinaryOperator):
            binary_count += 1
            total_children += 2
        elif isinstance(node, RollingOperator):
            rolling_count += 1
            total_children += 1
            window_values.append(float(node._delta_time))
        elif isinstance(node, PairRollingOperator):
            pair_rolling_count += 1
            total_children += 2
            window_values.append(float(node._delta_time))

        op_name = node.__class__.__name__
        if op_name in AST_OPERATOR_NAMES:
            op_counter[op_name] += 1
        if op_name in RISKY_OPERATOR_NAMES:
            risky_op_count += 1
        if isinstance(node, Div):
            div_count += 1
        elif op_name == "Log":
            log_count += 1
        elif isinstance(node, (Less, Greater)):
            less_greater_count += 1
        elif isinstance(node, (Corr, Cov)):
            corr_cov_count += 1

    operator_count = unary_count + binary_count + rolling_count + pair_rolling_count
    node_count_safe = max(1, node_count)
    avg_branching_factor = float(total_children) / max(1, operator_count)
    window_count = len(window_values)
    if window_values:
        window_arr = np.asarray(window_values, dtype=np.float32)
        window_min = float(window_arr.min())
        window_max = float(window_arr.max())
        window_mean = float(window_arr.mean())
        window_std = float(window_arr.std())
    else:
        window_min = 0.0
        window_max = 0.0
        window_mean = 0.0
        window_std = 0.0

    feat = OrderedDict()
    feat["node_count"] = float(node_count)
    feat["leaf_count"] = float(leaf_count)
    feat["max_depth"] = float(expression_depth(expr))
    feat["avg_branching_factor"] = float(avg_branching_factor)
    feat["window_count"] = float(window_count)
    feat["window_min"] = float(window_min)
    feat["window_max"] = float(window_max)
    feat["window_mean"] = float(window_mean)
    feat["window_std"] = float(window_std)
    feat["risky_op_count"] = float(risky_op_count)
    feat["div_count"] = float(div_count)
    feat["log_count"] = float(log_count)
    feat["less_greater_count"] = float(less_greater_count)
    feat["corr_cov_count"] = float(corr_cov_count)
    feat["feature_ratio"] = float(feature_count) / node_count_safe
    feat["constant_ratio"] = float(constant_count) / node_count_safe
    feat["unary_ratio"] = float(unary_count) / node_count_safe
    feat["binary_ratio"] = float(binary_count) / node_count_safe
    feat["rolling_ratio"] = float(rolling_count) / node_count_safe
    feat["pair_rolling_ratio"] = float(pair_rolling_count) / node_count_safe
    for name in AST_FEATURE_FIELDS:
        feat[f"field_ratio_{name}"] = float(field_counter.get(name, 0)) / node_count_safe
    for name in AST_OPERATOR_NAMES:
        feat[f"op_ratio_{name.lower()}"] = float(op_counter.get(name, 0)) / node_count_safe
    return feat


def extract_ast_feature_vector(expr: Expression) -> np.ndarray:
    feat_map = extract_ast_feature_map(expr)
    return np.asarray(list(feat_map.values()), dtype=np.float32)


def cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_norm = float(np.linalg.norm(lhs))
    rhs_norm = float(np.linalg.norm(rhs))
    if lhs_norm <= 1e-12 or rhs_norm <= 1e-12:
        return 0.0
    return float(np.dot(lhs, rhs) / (lhs_norm * rhs_norm))


def pairwise_cosine_similarity(vectors: Sequence[np.ndarray]) -> np.ndarray:
    n = len(vectors)
    sims = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(vectors[i], vectors[j])
            sims[i, j] = sim
            sims[j, i] = sim
    return sims


def connected_components_from_threshold(similarity: np.ndarray, threshold: float) -> List[int]:
    n = int(similarity.shape[0])
    if n == 0:
        return []
    labels = [-1] * n
    cluster_id = 0
    for start in range(n):
        if labels[start] >= 0:
            continue
        stack = [start]
        labels[start] = cluster_id
        while stack:
            idx = stack.pop()
            row = similarity[idx]
            for nxt in range(n):
                if labels[nxt] >= 0:
                    continue
                if idx == nxt or float(row[nxt]) >= threshold:
                    labels[nxt] = cluster_id
                    stack.append(nxt)
        cluster_id += 1
    return labels


def labels_to_members(labels: Sequence[int]) -> Dict[int, List[int]]:
    members: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        members.setdefault(int(label), []).append(idx)
    return members


def cluster_ast_features(exprs: Sequence[Expression], threshold: float) -> Tuple[List[int], np.ndarray, List[np.ndarray]]:
    vectors = [extract_ast_feature_vector(expr) for expr in exprs]
    if not vectors:
        return [], np.zeros((0, 0), dtype=np.float32), []
    similarity = pairwise_cosine_similarity(vectors)
    labels = connected_components_from_threshold(similarity, threshold)
    return labels, similarity, vectors


def cluster_output_correlations(exprs: Sequence[Expression], calculator, threshold: float) -> Tuple[List[int], np.ndarray]:
    n = len(exprs)
    if n == 0:
        return [], np.zeros((0, 0), dtype=np.float32)
    similarity = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            try:
                sim = abs(float(calculator.calc_mutual_IC(exprs[i], exprs[j])))
            except Exception:
                sim = 0.0
            if math.isnan(sim) or math.isinf(sim):
                sim = 0.0
            similarity[i, j] = sim
            similarity[j, i] = sim
    labels = connected_components_from_threshold(similarity, threshold)
    return labels, similarity
