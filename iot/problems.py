"""
Problem definitions for distributed constraint optimization.
Contains problem configurations and preset problems.
"""

import enum
import random
from dataclasses import dataclass
from typing import Callable, List

class ProblemID(enum.Enum):
    """Identifiers for preset problem types"""
    U_K025 = "Uniform_k0.25"
    U_K075 = "Uniform_k0.75"
    COLOR = "GraphColoring_k0.10"


@dataclass
class ProblemConfig:
    """configuration for a DCOP problem"""
    pid: ProblemID
    density: float
    domain: int
    cost_fn: Callable[[int, int], int]
    description: str = ""


def _uniform_cost(lb: int = 100, ub: int = 200) -> Callable[[int, int], int]:
    """Generate a uniform random cost function"""
    def fn(x: int, y: int) -> int:
        return random.randint(lb, ub)
    return fn


def _coloring_cost(lb: int = 100, ub: int = 200) -> Callable[[int, int], int]:
    """Generate a graph coloring cost function"""
    def fn(x: int, y: int) -> int:
        return random.randint(lb, ub) if x == y else 0
    return fn


# Preset problem configurations
PRESET_PROBLEMS: List[ProblemConfig] = [
    ProblemConfig(
        pid=ProblemID.U_K025,
        density=0.25,
        domain=5,
        cost_fn=_uniform_cost(),
        description="Uniform Random Graph (k=0.25, domain=5)"
    ),
    ProblemConfig(
        pid=ProblemID.U_K075,
        density=0.75,
        domain=5,
        cost_fn=_uniform_cost(),
        description="Uniform Random Graph (k=0.75, domain=5)"
    ),
    ProblemConfig(
        pid=ProblemID.COLOR,
        density=0.10,
        domain=3,
        cost_fn=_coloring_cost(),
        description="Graph Coloring Problem (k=0.10, domain=3)"
    ),
]