import math
import re
from collections import Counter
from typing import Counter as CounterType

from alphagen.data.expression import Expression


FIELD_PATTERN = re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*")
CONST_PATTERN = re.compile(r"Constant\((-?\d+(?:\.\d+)?)\)|(?<![A-Za-z_$])-?\d+(?:\.\d+)?")


def ast_signature(expr: Expression) -> str:
    """Return a coarse AST signature that removes concrete fields/constants."""
    text = str(expr)
    text = FIELD_PATTERN.sub("FIELD", text)
    text = CONST_PATTERN.sub("CONST", text)
    return text


class AstCountIntrinsic:
    def __init__(self) -> None:
        self.counts: CounterType[str] = Counter()

    def reward(self, signature: str) -> float:
        n = self.counts.get(signature, 0)
        return 1.0 / math.sqrt(n + 1.0)

    def update(self, signature: str) -> None:
        self.counts[signature] += 1

    def to_dict(self) -> dict:
        return dict(self.counts)

