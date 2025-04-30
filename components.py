from typing import Callable, Dict, Tuple
import numpy as np
from numpy import ndarray

from agent import Agent

class Message:
    def __init__(self, sender, receiver, content):
        self.sender = sender
        self.receiver = receiver
        self.content = content

class CostTable:
    def __init__(self,agents:Tuple[Agent,Agent],domain:int,ct_creation:Callable,**kwargs):
        self.table = self._create_table(domain,ct_creation,**kwargs)
        self.connections = {agent: i for i,agent in enumerate(agents)}
    def _create_table(self,domain:int,ct_creation:Callable,**kwargs) -> ndarray:
        return ct_creation(size=(domain,)*2,**kwargs)
