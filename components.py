from typing import Callable, Dict, Tuple

import np

from agent import Agent
from utils import create_table


class Message:
    def __init__(self, sender, receiver, content):
        self.sender = sender
        self.receiver = receiver
        self.content = content

class CostTable:
    def __init__(self,domain:int,ct_creation:Callable,agents:Tuple[Agent],**kwargs):
        self.table = create_table(domain,ct_creation,**kwargs)
        self.connections = {agent: i for i,agent in enumerate(agents)}
