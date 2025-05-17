"""
Experiment runner for distributed constraint optimization.
Contains classes and functions for running experiments and visualizing results.
"""

import logging
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from iot.core import Agent
from iot.agents import DSA, MGM, MGM2
from iot.graph import generate_constraint_graph
from iot.problems import ProblemConfig, PRESET_PROBLEMS

# Logging config
logger = logging.getLogger(__name__)

@dataclass
class ExperimentRunner:
    """Runner for DCOP algorithm experiments"""
    rounds: int
    runs: int = 30                # 30 graphs per problem
    visualize: bool = False       # node‑level visuals off - only for the word document
    save_plots: bool = True
    output_dir: Path = Path.cwd() / "plots"

    def __post_init__(self):
        if self.save_plots:
            self.output_dir.mkdir(exist_ok=True, parents=True)

    # ------------------------------------------------------------------
    def _calculate_global_cost(self, agents: List[Agent], mailer) -> int:
        """Calculate the total cost across all constraints in the graph"""
        total_cost = 0
        seen_pairs: Set[Tuple[str, str]] = set()
        for a in agents:
            for n in a.neighbors:
                pair = tuple(sorted([a.name, n]))
                if pair not in seen_pairs:
                    total_cost += a.constraints[n].cost(a.assignment, mailer.global_mapping[n].assignment)
                    seen_pairs.add(pair)
        return total_cost

    # ------------------------------------------------------------------
    def _run_single(self, seed: int, prob: ProblemConfig, algo_cls: type[Agent]) -> List[int]:
        """Run a single experiment with the given seed, problem, and algorithm"""
        random.seed(seed)
        np.random.seed(seed)

        agents, _graph, mailer = generate_constraint_graph(
            num_agents=30,
            density=prob.density,
            domain_size=prob.domain,
            cost_fn=prob.cost_fn,
            agent_class=algo_cls
        )

        for a in agents:
            for n in a.neighbors:
                a._send(n, np.array([a.assignment]), "assignment")
        mailer.deliver_all()

        curve = [self._calculate_global_cost(agents, mailer)]

        for _ in range(self.rounds):
            for a in agents:
                a.decide()
            mailer.deliver_all()
            curve.append(self._calculate_global_cost(agents, mailer))
            for a in agents:
                a.after_round()

        return curve

    # ------------------------------------------------------------------
    def _avg_runs(self, prob: ProblemConfig, algo_cls: type[Agent]) -> List[float]:
        """Run multiple experiments and average the results"""
        all_curves: List[List[int]] = []
        logger.info(f"Running {algo_cls.__name__} on {prob.pid.value} ({self.runs} graphs)")

        for seed in range(self.runs):
            all_curves.append(self._run_single(seed, prob, algo_cls))
            if (seed + 1) % 10 == 0:
                logger.info(f"  {seed + 1}/{self.runs} graphs done")

        return [statistics.mean(c) for c in zip(*all_curves)]

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Run experiments for all algorithms on all preset problems"""
        plt.style.use("seaborn-v0_8-darkgrid")
        mpl.rcParams["font.family"] = "DejaVu Sans"

        algorithms = {
            "DSA (p=0.2)": type("DSA_p0_2", (DSA,), {"p": 0.2}),
            "DSA (p=0.7)": type("DSA_p0_7", (DSA,), {"p": 0.7}),
            "DSA (p=1.0)": type("DSA_p1_0", (DSA,), {"p": 1.0}),
            "MGM": MGM,
            "MGM-2": MGM2,
        }

        colours = {
            "DSA (p=0.2)": "#1f77b4",
            "DSA (p=0.7)": "#ff7f0e",
            "DSA (p=1.0)": "#2ca02c",
            "MGM": "#d62728",
            "MGM-2": "#9467bd",
        }
        styles = {
            "DSA (p=0.2)": "-",
            "DSA (p=0.7)": "-",
            "DSA (p=1.0)": "-",
            "MGM": "--",
            "MGM-2": "-.",
        }

        for prob in PRESET_PROBLEMS:
            plt.figure(figsize=(12, 7))
            for name, cls in algorithms.items():
                avg = self._avg_runs(prob, cls)
                plt.plot(avg,
                         label=f"{name} (final: {avg[-1]:.1f})",
                         color=colours[name],
                         linestyle=styles[name],
                         linewidth=2.5)

            plt.title(f"Algorithm Comparison – {prob.description}", fontsize=16)
            plt.xlabel("Iteration", fontsize=14)
            plt.ylabel("Average Global Cost", fontsize=14)
            plt.legend(fontsize=12)
            plt.tight_layout()

            out_path = self.output_dir / f"{prob.pid.value}.png"
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved {out_path}")
            plt.close()
