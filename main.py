import enum
import logging
import random
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Set

import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import numpy as np

# Logging config
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d – %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# core
@dataclass
class Message:
    """A message exchanged between agents"""
    data: np.ndarray
    sender: "Agent"
    recipient: "Agent"
    msg_type: str = "value"  # "value", "gain", "offer", "accept"

    def copy(self) -> "Message":
        return Message(self.data.copy(), self.sender, self.recipient, self.msg_type)


class Mailer:
    """Central mailer for sending messages between agents"""

    def __init__(self) -> None:
        self._queue: List[Message] = []

    def post(self, msg: Message) -> None:
        self._queue.append(msg)

    def deliver_all(self) -> None:
        for msg in self._queue:
            msg.recipient.inbox.append(msg)
        self._queue.clear()


@dataclass
class ConstraintCost:
    """A constraint between two agents with associated costs"""
    agent1: str
    agent2: str
    cost_matrix: np.ndarray  # matrix[a1_val][a2_val] = cost

    def cost(self, a1_val: int, a2_val: int) -> int:
        if a1_val < 0 or a2_val < 0:
            return 0
        return int(self.cost_matrix[a1_val, a2_val])


class Agent:
    """Base agent class for DCOP algorithms"""

    def __init__(self, name: str, domain_size: int):
        self.name = name
        self.domain_size = domain_size
        self.value = random.randrange(domain_size)
        self.neighbours: List["Agent"] = []
        self.constraints: Dict[str, ConstraintCost] = {}
        self.inbox: List[Message] = []
        self.mailer: Optional[Mailer] = None

    # ---------- Graph utilities ------------------------------------------------

    def connect(self, other: "Agent", cost_fn: Callable[[int, int], int]) -> None:
        if other not in self.neighbours:
            self.neighbours.append(other)
            other.neighbours.append(self)

            cost_matrix = np.zeros((self.domain_size, other.domain_size), dtype=int)
            for i in range(self.domain_size):
                for j in range(other.domain_size):
                    cost_matrix[i, j] = cost_fn(i, j)

            constraint = ConstraintCost(self.name, other.name, cost_matrix)
            self.constraints[other.name] = constraint
            # reverse
            other.constraints[self.name] = ConstraintCost(other.name, self.name, cost_matrix.T)

    # ---------- Messaging ------------------------------------------------------

    def _send(self, recipient: "Agent", data: np.ndarray, msg_type: str = "value") -> None:
        assert self.mailer, "Mailer not attached"
        self.mailer.post(Message(data, self, recipient, msg_type))

    # ---------- DCOP  methods  -------------------------------------------------------

    def decide(self) -> None:
        raise NotImplementedError

    def after_round(self) -> None:
        self.inbox.clear()

    def _cost(self, value: int, assignments: Dict[str, int]) -> int:
        total_cost = 0
        for n in self.neighbours:
            n_value = assignments.get(n.name, n.value)
            constraint = self.constraints[n.name]
            total_cost += constraint.cost(value, n_value)
        return total_cost

    def current_local_cost(self) -> int:
        assignments = {n.name: n.value for n in self.neighbours}
        return self._cost(self.value, assignments)

# Algorithms
class DSAC(Agent):
    p: float = 0.7

    def decide(self) -> None:
        assignments = {m.sender.name: int(m.data[0]) for m in self.inbox if m.msg_type == "value"}

        cur_cost = self._cost(self.value, assignments)
        best_val, best_cost = self.value, cur_cost

        for v in range(self.domain_size):
            if v == self.value:
                continue
            c = self._cost(v, assignments)
            if c < best_cost:
                best_cost, best_val = c, v

        if best_cost < cur_cost and random.random() < self.p:
            self.value = best_val

        for n in self.neighbours:
            self._send(n, np.array([self.value]), "value")


class MGM(Agent):
    def __init__(self, name: str, domain_size: int):
        super().__init__(name, domain_size)
        self.mode = "value"
        self.best_gain = 0
        self.best_val = self.value

    def decide(self) -> None:
        if self.mode == "value":
            for n in self.neighbours:
                self._send(n, np.array([self.value]), "value")
            self.mode = "gain"
            return

        if self.mode == "gain":
            assignments = {m.sender.name: int(m.data[0]) for m in self.inbox if m.msg_type == "value"}
            cur_cost = self._cost(self.value, assignments)
            self.best_gain, self.best_val = 0, self.value
            for v in range(self.domain_size):
                if v == self.value:
                    continue
                gain = cur_cost - self._cost(v, assignments)
                if gain > self.best_gain:
                    self.best_gain, self.best_val = gain, v

            for n in self.neighbours:
                self._send(n, np.array([self.best_gain]), "gain")
            self.mode = "select"
            return

        if self.mode == "select":
            gains = [int(m.data[0]) for m in self.inbox if m.msg_type == "gain"]
            if self.best_gain > 0 and all(self.best_gain > g for g in gains):
                self.value = self.best_val
            for n in self.neighbours:
                self._send(n, np.array([self.value]), "value")
            self.mode = "value"


class MGM2(Agent):
    """MGM‑2 with the ‘self‑gain’ bug fixed"""

    def __init__(self, name: str, domain_size: int):
        super().__init__(name, domain_size)
        self.is_offerer = False
        self.current_partner: Optional[str] = None
        self.best_offer: Optional[Tuple[int, int, int]] = None  # (partner_val, my_val, gain)
        self.mode = "value"
        self.best_gain: int = 0
        self.best_val: int = self.value

    # ----------------------------------------------------------------------
    def decide(self) -> None:
        #phase 1: broadcast value
        if self.mode == "value":
            self.is_offerer = random.random() < 0.5
            self.current_partner = None
            self.best_offer = None

            for n in self.neighbours:
                self._send(n, np.array([self.value]), "value")

            self.mode = "compute_gain"
            return

        #phase 2: compute best gains
        if self.mode == "compute_gain":
            assignments = {m.sender.name: int(m.data[0]) for m in self.inbox if m.msg_type == "value"}
            for neighbor in self.neighbours:
                assignments.setdefault(neighbor.name, neighbor.value)

            cur_cost = self._cost(self.value, assignments)
            best_val, best_gain = self.value, 0

            for v in range(self.domain_size):
                if v == self.value:
                    continue
                gain = cur_cost - self._cost(v, assignments)
                if gain > best_gain:
                    best_gain, best_val = gain, v

            # keep for next phase
            self.best_gain = best_gain
            self.best_val = best_val

            # bilateral search if I’m an offerer
            if self.is_offerer:
                for neighbor in self.neighbours:
                    n_name = neighbor.name
                    n_val = assignments[n_name]

                    for my_val in range(self.domain_size):
                        for n_new_val in range(neighbor.domain_size):
                            if my_val == self.value and n_new_val == n_val:
                                continue

                            old_cost = self._cost(self.value, assignments) + \
                                       neighbor._cost(n_val, {self.name: self.value, **assignments})

                            mod_asg = assignments.copy()
                            mod_asg[n_name] = n_new_val
                            new_cost = self._cost(my_val, mod_asg) + \
                                       neighbor._cost(n_new_val, {self.name: my_val, **mod_asg})

                            gain = old_cost - new_cost
                            if gain > best_gain:
                                best_gain = gain
                                self.best_offer = (n_new_val, my_val, gain)
                                self.current_partner = n_name
                                self.best_gain = best_gain  # update

            # broadcast *my* best gain
            for n in self.neighbours:
                self._send(n, np.array([self.best_gain]), "gain")

            self.mode = "process_gains"
            return

        #phase 3: compare gains / make offers
        if self.mode == "process_gains":
            gains_received = {m.sender.name: int(m.data[0])
                              for m in self.inbox if m.msg_type == "gain"}

            # bilateral offer
            if self.is_offerer and self.best_offer and self.current_partner:
                partner_gain = gains_received.get(self.current_partner, 0)
                if self.best_gain > partner_gain and self.best_gain > 0:
                    partner = next(n for n in self.neighbours if n.name == self.current_partner)
                    self._send(partner, np.array(self.best_offer), "offer")
                    self.mode = "finalize"
                    return

            #unilateral move
            best_external_gain = max(gains_received.values(), default=0)
            if self.best_gain > 0 and self.best_gain >= best_external_gain:
                self.value = self.best_val

            for n in self.neighbours:
                self._send(n, np.array([self.value]), "value")

            self.mode = "finalize"
            return

        # phase 4: finalize
        if self.mode == "finalize":
            if not self.is_offerer:
                offers = [(m.sender.name, m.data) for m in self.inbox if m.msg_type == "offer"]
                if offers:
                    sender_name, offer_data = max(offers, key=lambda o: o[1][2])
                    my_new_val, gain = int(offer_data[0]), int(offer_data[2])
                    if gain > 0:
                        partner = next(n for n in self.neighbours if n.name == sender_name)
                        self._send(partner, np.array([1]), "accept")
                        self.value = my_new_val

            else:
                if self.best_offer and self.current_partner:
                    accepted = any(m.msg_type == "accept" and
                                   m.sender.name == self.current_partner
                                   for m in self.inbox)
                    if accepted:
                        self.value = self.best_offer[1]

            # reset
            self.mode = "value"
            self.is_offerer = False
            self.current_partner = None
            self.best_offer = None
            self.best_gain = 0

            for n in self.neighbours:
                self._send(n, np.array([self.value]), "value")

# Problem definitions
class ProblemID(enum.Enum):
    U_K025 = "Uniform_k0.25"
    U_K075 = "Uniform_k0.75"
    COLOR = "GraphColoring_k0.10"


@dataclass
class ProblemConfig:
    pid: ProblemID
    density: float
    domain: int
    cost_fn: Callable[[int, int], int]
    description: str = ""


def _uniform_cost(lb: int = 100, ub: int = 200) -> Callable[[int, int], int]:
    def fn(x: int, y: int) -> int:
        return random.randint(lb, ub)
    return fn


def _coloring_cost(lb: int = 100, ub: int = 200) -> Callable[[int, int], int]:
    def fn(x: int, y: int) -> int:
        return random.randint(lb, ub) if x == y else 0
    return fn


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

# ----------------------------------------------------------------------------
# Graph generation -----------------------------------------------------------
# ----------------------------------------------------------------------------
def generate_constraint_graph(num_agents: int, density: float, domain_size: int,
                              cost_fn: Callable[[int, int], int],
                              agent_class: type[Agent]) -> Tuple[List[Agent], nx.Graph, Mailer]:

    mailer = Mailer()
    agents = [agent_class(f"A{i}", domain_size) for i in range(num_agents)]
    for a in agents:
        a.mailer = mailer

    while True:
        # generate a random graph with the given density
        G = nx.erdos_renyi_graph(num_agents, density)
        if nx.is_connected(G):
            break

    for i, j in G.edges():
        agents[i].connect(agents[j], cost_fn)

    # basic nx graph just for optional visualisation
    graph = nx.Graph()
    for i, a in enumerate(agents):
        graph.add_node(a.name, value=a.value, pos=(i % 5, i // 5))
    for a in agents:
        for n in a.neighbours:
            graph.add_edge(a.name, n.name)

    return agents, graph, mailer

# runner
@dataclass
class ExperimentRunner:
    rounds: int
    runs: int = 50                # 50 graphs per problem
    visualize: bool = False       # node‑level visuals off – keeps disk clean
    save_plots: bool = True
    output_dir: Path = Path.cwd() / "plots"

    def __post_init__(self):
        if self.save_plots:
            self.output_dir.mkdir(exist_ok=True, parents=True)

    # ------------------------------------------------------------------
    def _calculate_global_cost(self, agents: List[Agent]) -> int:
        total_cost = 0
        seen_pairs: Set[Tuple[str, str]] = set()
        for a in agents:
            for n in a.neighbours:
                pair = tuple(sorted([a.name, n.name]))
                if pair not in seen_pairs:
                    total_cost += a.constraints[n.name].cost(a.value, n.value)
                    seen_pairs.add(pair)
        return total_cost

    # ------------------------------------------------------------------
    def _run_single(self, seed: int, prob: ProblemConfig, algo_cls: type[Agent]) -> List[int]:
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
            for n in a.neighbours:
                a._send(n, np.array([a.value]), "value")
        mailer.deliver_all()

        curve = [self._calculate_global_cost(agents)]

        for _ in range(self.rounds):
            for a in agents:
                a.decide()
            mailer.deliver_all()
            curve.append(self._calculate_global_cost(agents))
            for a in agents:
                a.after_round()

        return curve

    # ------------------------------------------------------------------
    def _avg_runs(self, prob: ProblemConfig, algo_cls: type[Agent]) -> List[float]:
        all_curves: List[List[int]] = []
        logger.info(f"Running {algo_cls.__name__} on {prob.pid.value} ({self.runs} graphs)")

        for seed in range(self.runs):
            all_curves.append(self._run_single(seed, prob, algo_cls))
            if (seed + 1) % 10 == 0:
                logger.info(f"  {seed + 1}/{self.runs} graphs done")

        return [statistics.mean(c) for c in zip(*all_curves)]

    # ------------------------------------------------------------------
    def run(self) -> None:
        plt.style.use("seaborn-v0_8-darkgrid")
        mpl.rcParams["font.family"] = "DejaVu Sans"

        algorithms = {
            "DSA-C (p=0.2)": type("DSA_p0_2", (DSAC,), {"p": 0.2}),
            "DSA-C (p=0.7)": type("DSA_p0_7", (DSAC,), {"p": 0.7}),
            "DSA-C (p=1.0)": type("DSA_p1_0", (DSAC,), {"p": 1.0}),
            "MGM": MGM,
            "MGM-2": MGM2,
        }

        colours = {
            "DSA-C (p=0.2)": "#1f77b4",
            "DSA-C (p=0.7)": "#ff7f0e",
            "DSA-C (p=1.0)": "#2ca02c",
            "MGM": "#d62728",
            "MGM-2": "#9467bd",
        }
        styles = {
            "DSA-C (p=0.2)": "-",
            "DSA-C (p=0.7)": "-",
            "DSA-C (p=1.0)": "-",
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

# ----------------------------------------------------------------------------
# Main -----------------------------------------------------------------------
# ----------------------------------------------------------------------------
def main() -> None:
    runner = ExperimentRunner(rounds=50)
    runner.run()

if __name__ == "__main__":
    main()
