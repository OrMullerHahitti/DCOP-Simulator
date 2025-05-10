"""
Core components for distributed constraint optimization.
Contains base classes for messaging and agent functionality.
"""

import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable

import numpy as np

# Logging config
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d – %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

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
        self.value = random.randrange(domain_size) #initiate random assignment
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
    #abstract
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