"""
Core components for distributed constraint optimization.
Contains base classes for messaging and agent functionality.
"""

import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable

import numpy as np

# Logging config- for debugging
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d – %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

@dataclass  # the object doesnt do anything so we can use dataclass
class Message:
    """A message exchanged between agents"""
    data: np.ndarray
    sender: str
    recipient: str
    msg_type: str = "assignment"  # "assignment", "lr", "offer", "accept"

    def copy(self) -> "Message":  # deep copy- new message object
        return Message(self.data.copy(), self.sender, self.recipient, self.msg_type)




class Agent:
    """Base agent class for DCOP algorithms"""

    def __init__(self, name: str, domain_size: int):
        self.name = name
        self.domain_size = domain_size
        self.assignment = random.randrange(domain_size) #initiate random assignment
        self.neighbors: Dict[str, int] = {}
        self.constraints: Dict[str, ConstraintCost] = {}
        self.inbox: List[Message] = []
        self.mailer: Optional[Mailer] = None

    # ---------- Graph utilities ------------------------------------------------

    def connect(self, other: "Agent", cost_fn: Callable[[int, int], int]) -> None:  # create a neighbour relationship
        if other not in self.neighbors:
            self.neighbors[other.name] = None
            other.neighbors[self.name] = None

            cost_matrix = np.zeros((self.domain_size, other.domain_size), dtype=int)  # creating a cost matrix for the constraint according to the problems specifications
            for i in range(self.domain_size):
                for j in range(other.domain_size):
                    cost_matrix[i, j] = cost_fn(i, j)

            constraint = ConstraintCost(self.name, other.name, cost_matrix)
            self.constraints[other.name] = constraint
            # reverse
            other.constraints[self.name] = ConstraintCost(other.name, self.name, cost_matrix.T)  # transposed so every agent will be represented by the rows in the cost matrix

    # ---------- Messaging ------------------------------------------------------

    def _send(self, recipient: str, data: np.ndarray, msg_type: str = "assignment") -> None:
        assert self.mailer, "Mailer not attached"  # check if mailer is attached, if nt it means the problem was not initialized yet
        self.mailer.post(Message(data, self.name, recipient, msg_type))

    def update_neighbors_assignments(self, neighbors_assignments: Dict[str, int]) -> None:
        for n in neighbors_assignments:
            if n in self.neighbors:
                self.neighbors[n] = neighbors_assignments[n]  # update the assignment of the neighbor

    # ---------- DCOP  methods  -------------------------------------------------------
    #abstract
    def decide(self) -> None:  # one iteration of the algorithm
        raise NotImplementedError

    def after_round(self) -> None:
        self.inbox.clear()

    def _get_local_cost(self, assignment: int) -> int:  # calculate the local cost of an assignment
        total_cost = 0
        for n in self.neighbors:
            n_assignment = self.neighbors[n]  # returns the assignment of the neighbor
            constraint = self.constraints[n]
            total_cost += constraint.cost(assignment, n_assignment)
        return total_cost

    def current_local_cost(self) -> int:
        return self._get_local_cost(self.assignment)


class Mailer:
    """Central mailer for sending messages between agents"""

    def __init__(self, agents: List[Agent]) -> None:
        self._queue: List[Message] = []
        self.agents = agents
        self.global_mapping = {a.name: a for a in self.agents}  # mapping of agent names to agent objects

    def post(self, msg: Message) -> None:  # take a message from an agent and add it to the queue
        self._queue.append(msg)

    def deliver_all(self) -> None:
        for msg in self._queue:
            recipient = self.global_mapping[msg.recipient]
            recipient.inbox.append(msg)
        self._queue.clear()


@dataclass
class ConstraintCost:
    """A constraint between two agents with associated costs"""
    agent1: str
    agent2: str
    cost_matrix: np.ndarray  # matrix[a1_assignment][a2_assignment] = cost

    def cost(self, a1_assignment: int, a2_assignment: int) -> int:
        if a1_assignment < 0 or a2_assignment < 0:
            return 0
        return int(self.cost_matrix[a1_assignment, a2_assignment])

