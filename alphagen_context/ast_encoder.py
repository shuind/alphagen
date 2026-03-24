from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from torch import Tensor, nn

from alphagen.data.expression import (
    BinaryOperator,
    Constant,
    Expression,
    Feature,
    PairRollingOperator,
    RollingOperator,
    UnaryOperator,
)

try:
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import GATConv, SAGEConv, global_mean_pool
except ImportError as exc:  # pragma: no cover
    Batch = None  # type: ignore[assignment]
    Data = None  # type: ignore[assignment]
    GATConv = None  # type: ignore[assignment]
    SAGEConv = None  # type: ignore[assignment]
    global_mean_pool = None  # type: ignore[assignment]
    _PYG_IMPORT_ERROR = exc
else:
    _PYG_IMPORT_ERROR = None


NODE_TYPE_OPERATOR = 0
NODE_TYPE_FEATURE = 1
NODE_TYPE_CONSTANT = 2
NODE_TYPE_PARAMETER = 3


def _require_pyg() -> None:
    if _PYG_IMPORT_ERROR is not None:
        raise RuntimeError(
            "torch-geometric is required for alphagen_context AST-GNN. "
            "Install with `pip install torch-geometric==2.7.0`."
        ) from _PYG_IMPORT_ERROR


@dataclass
class ASTNodeRecord:
    node_type: int
    attr_id: int
    scalar: float


class ASTGraphBuilder:
    def __init__(self, attr_vocab_size: int = 256):
        _require_pyg()
        self.attr_vocab_size = attr_vocab_size

    def _stable_id(self, key: str) -> int:
        return abs(hash(key)) % self.attr_vocab_size

    def build(self, expr: Expression) -> Data:
        records: List[ASTNodeRecord] = []
        edges: List[List[int]] = []

        def visit(node: Expression, parent_idx: int | None = None) -> None:
            idx = len(records)
            records.append(self._encode_node(node))
            if parent_idx is not None:
                edges.append([parent_idx, idx])
                edges.append([idx, parent_idx])
            for child in self._children(node):
                visit(child, idx)

        visit(expr)
        x_type = torch.tensor([record.node_type for record in records], dtype=torch.long)
        x_attr = torch.tensor([record.attr_id for record in records], dtype=torch.long)
        x_scalar = torch.tensor([[record.scalar] for record in records], dtype=torch.float32)
        edge_index = (
            torch.tensor(edges, dtype=torch.long).t().contiguous()
            if edges
            else torch.zeros((2, 0), dtype=torch.long)
        )
        return Data(
            x_type=x_type,
            x_attr=x_attr,
            x_scalar=x_scalar,
            edge_index=edge_index,
            num_nodes=len(records),
        )

    def build_batch(self, exprs: Sequence[Expression]) -> Batch:
        return Batch.from_data_list([self.build(expr) for expr in exprs])

    def _encode_node(self, node: Expression) -> ASTNodeRecord:
        if isinstance(node, Feature):
            return ASTNodeRecord(
                node_type=NODE_TYPE_FEATURE,
                attr_id=int(node._feature.value),
                scalar=float(node._feature.value),
            )
        if isinstance(node, Constant):
            return ASTNodeRecord(
                node_type=NODE_TYPE_CONSTANT,
                attr_id=self._stable_id("constant"),
                scalar=float(node._value),
            )
        if isinstance(node, (RollingOperator, PairRollingOperator)):
            return ASTNodeRecord(
                node_type=NODE_TYPE_OPERATOR,
                attr_id=self._stable_id(type(node).__name__),
                scalar=float(node._delta_time),
            )
        if isinstance(node, (UnaryOperator, BinaryOperator)):
            return ASTNodeRecord(
                node_type=NODE_TYPE_OPERATOR,
                attr_id=self._stable_id(type(node).__name__),
                scalar=0.0,
            )
        return ASTNodeRecord(
            node_type=NODE_TYPE_PARAMETER,
            attr_id=self._stable_id(type(node).__name__),
            scalar=0.0,
        )

    def _children(self, node: Expression) -> List[Expression]:
        if isinstance(node, UnaryOperator):
            return [node._operand]
        if isinstance(node, BinaryOperator):
            return [node._lhs, node._rhs]
        if isinstance(node, RollingOperator):
            return [node._operand, Constant(float(node._delta_time))]
        if isinstance(node, PairRollingOperator):
            return [node._lhs, node._rhs, Constant(float(node._delta_time))]
        return []


class ASTGraphEncoder(nn.Module):
    def __init__(
        self,
        attr_vocab_size: int = 256,
        hidden_dim: int = 64,
        output_dim: int = 32,
        conv_type: str = "sage",
        dropout: float = 0.1,
    ):
        _require_pyg()
        super().__init__()
        self.output_dim = output_dim
        self.node_type_emb = nn.Embedding(4, 8)
        self.attr_emb = nn.Embedding(attr_vocab_size, 8)
        self.scalar_proj = nn.Linear(1, 8)
        self.input_proj = nn.Linear(24, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        conv_cls = GATConv if conv_type == "gat" else SAGEConv
        self.conv1 = conv_cls(hidden_dim, hidden_dim)
        self.conv2 = conv_cls(hidden_dim, hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, batch_graph: Batch) -> Tensor:
        x = torch.cat(
            [
                self.node_type_emb(batch_graph.x_type),
                self.attr_emb(batch_graph.x_attr),
                self.scalar_proj(batch_graph.x_scalar),
            ],
            dim=1,
        )
        x = self.input_proj(x)
        x = torch.relu(self.conv1(x, batch_graph.edge_index))
        x = self.dropout(x)
        x = torch.relu(self.conv2(x, batch_graph.edge_index))
        pooled = global_mean_pool(x, batch_graph.batch)
        return self.output_proj(pooled)
