from __future__ import annotations
from typing import Protocol, Tuple, List, Dict, Callable
from typing import TypeAlias

from numpy import ndarray

lr: TypeAlias = float| int
assignment: TypeAlias = int| float




class AgentGraph(Protocol):
    def __init__(self, agents: list[Agent]):
        self.agents = agents
        self.iteration = 0
        self.global_cost = float('inf')


class CostTable(Protocol):
    def __init__(self, domain: int, ct_creation: Callable, agents: Tuple[Agent], **kwargs):
        self.table = ndarray
        self.connections = Dict[Agent, int]

class Message(Protocol):
    def __init__(self, sender: Agent, receiver: Agent, content: assignment|lr):
        self.sender = sender
        self.receiver = receiver
        self.content = content

class Agent(Protocol):
    def __init__(self,id:int ,domain_size:int):
        self.id=id
        self.mailbox : List[Message] = []
        self.domain = domain_size
        self.assignment : int = np.random.randint(low=0,high=domain_size)
        self.neighbours : Dict[Agent,CostTable] = {}
        self.neighbours_assignments : Dict[Agent,assignment] = {}
        self.local_cost = float('inf')

    def receive_message(self, message:Message):
        pass
    def find_assignment(self)-> int:
        pass

class Method(Protocol):
    def __call__(self, *args, **kwargs)->Tuple[assignment,lr]:
        pass




