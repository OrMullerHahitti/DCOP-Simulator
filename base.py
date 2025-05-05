from __future__ import annotations
import random
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Callable, Tuple, Optional
import matplotlib.pyplot as plt
import networkx as nx

import numpy as np


class MessageType(Enum):
    LR = "LR"
    ASSIGNMENT = "ASSIGNMENT"

    def __str__(self):
        return self.value


class Message:
    def __init__(self, sender: str, recipient: str, content):
        self.type = None
        self.sender = sender
        self.recipient = recipient
        self.content = content


class LRMessage(Message):
    def __init__(self, sender: str, receiver: str, content: Tuple[int, float]):
        super().__init__(sender, receiver, content)
        self.type = MessageType.LR
        self.content = content
        self.sender_assignment = content[0]
        self.sender_lr = content[1]


class AssignmentMessage(Message):
    def __init__(self, sender: str, receiver: str, content: int):
        super().__init__(sender, receiver, content)
        self.type = MessageType.ASSIGNMENT
        self.content = content
        self.sender_assignment = content


class CostTable:
    def __init__(
        self,
        agents: Tuple[Agent, Agent],
        domain: int,
        ct_creation: Callable,
        ct_kwargs: Dict,
    ):
        self.table = self._create_table(domain, ct_creation, ct_kwargs)
        self.connections = {agent: i for i, agent in enumerate(agents)}

    @staticmethod
    def _create_table(domain: int, ct_creation: Callable, kwargs) -> np.ndarray:
        return ct_creation(size=((domain,) * 2), **kwargs)

    def shuffle(self):
        """
        Shuffle the cost table.
        """
        flattened = self.table.flatten()
        np.random.shuffle(flattened)
        return flattened.reshape(self.table.shape)


class Agent(ABC):
    def __init__(self, id: int, domain_size: int):
        self.id = id
        self.mailbox: List[Message] = []
        self.domain = domain_size
        self.assignment = np.random.randint(low=0, high=domain_size)
        self.neighbours: Dict[Agent, CostTable] = {}
        self.local_cost = float("inf")

    def receive_message(self, message: Message):
        self.mailbox.append(message)
    def send_messages(self):
        """
        Send messages to all neighbours.
        """

    ########################################

    def find_assignment(self) -> int:
        """
        Find the assignment for the agent based on the received messages.
        This method should be implemented by subclasses.
        """
        pass

    @property
    def name(self):
        return f"Agent_{self.id}"

    def empty_mailbox(self):
        self.mailbox = []
        # all dunder functions

    def __str__(self):
        return f"Agent({self.name})"
        # how the class will be shown in debugging or while editing

    def __repr__(self):
        return f"Agent({self.name})"

    def __eq__(self, other):
        if not isinstance(other, Agent):
            return False
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)


class AgentGraph:
    def __init__(
        self,
        num_variables: int,
        domain_size: int,
        density: float,
        ct_creation: Callable,#np.random.randint, etc
        ct_kwargs: Dict,#low, high, etc
    ):
        self.agents = self._create_agents(num_variables, domain_size)
        self.G = self._create_random_connected_graph(density)
        self.domain_size = domain_size
        self.iteration = 0
        self.global_cost = float("inf")
        self.mailer = Mailer()
        self.ct_creation = ct_creation
        self.ct_kwargs = ct_kwargs
        self._set_constraints()
        self.agents = list(self.G.nodes())

    def _set_constraints(self):
        for edge in self.G.edges():
            agent1, agent2 = edge
            ct = CostTable(
                (agent1, agent2), self.domain_size, self.ct_creation, self.ct_kwargs
            )
            agent1.neighbours[agent2.name] = ct
            agent2.neighbours[agent1.name] = ct

    def _create_agents(self, num_variables: int, domain_size: int):
        """
        Create agents with random assignments.
        """
        agents: List[Agent] = [Agent(i, domain_size) for i in range(num_variables)]
        return agents

    def _create_random_connected_graph(self, density: float):
        n = len(self.agents)
        # keep trying until you get a connected graph
        while True:
            G_int = nx.erdos_renyi_graph(n, density)
            if nx.is_connected(G_int):
                break
        # map 0→Agent_0, 1→Agent_1, …
        mapping = {i: self.agents[i] for i in range(n)}
        G = nx.relabel_nodes(G_int, mapping)
        return G

    def visualize(self):
        """
        Visualize the graph using matplotlib.
        """
        pos = nx.spring_layout(self.G)
        nx.draw(
            self.G,
            pos,
            with_labels=True,
            node_size=700,
            node_color="lightblue",
            font_size=10,
        )
        nx.draw_networkx_edge_labels(self.G, pos)
        plt.show()


array = np.random.randint(low=0, high=100, size=(5, 5))
array[:, 0]


class MGMAgent(Agent):
    def __init__(self, id: int, name: str, domain_size: int):
        super.__init__(id, name, domain_size)

    def create_new_messages(self) -> List[Message]:
        pass


class Mailer:
    def __init__(self):
        self.global_cost = float("inf")
        self.global_assignment: Dict[Agent, int] = {}
        self.mailbox: List[Message] = []
        self.agents: Dict[str:Agent] = {}

    def set_agents(self):
        """
        Set the agents in the mailbox.
        """
        for agent in self.agents.values():
            self.agents[agent.name] = agent

    def collect_messages(self):
        for agent in self.agents.values():
            self.mailbox.extend(agent.mailbox)
            agent.empty_mailbox()

    def send_messages(self):
        for message in self.mailbox:
            recipient = self.agents[message.recipient]
            recipient.receive_message(message)
