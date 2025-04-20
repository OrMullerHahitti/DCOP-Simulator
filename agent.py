from abc import ABC, abstractmethod
from typing import List, Dict, Callable

import numpy as np

from components import Message, CostTable


class Agent(ABC):
    def __init__(self,id:int ,name:str,domain_size:int):
        self.name = name
        self.id=id
        self.mailbox : List[Message] = []
        self.domain = domain_size
        self.assignment = None
        self.neighbours : Dict[Agent,CostTable] = {}
        self.local_cost = float('inf')


    def receive_message(self, message:Message):
        self.mailbox.append(message)

    ########################################
    def find_assignment(self)-> int:
        temp_cost = float('inf')
        for message in self.mailbox:
            temp_cost = np.min(temp_cost,message.content,axis=)




    def set_neighbors(self, neighbors:List[tuple]):
        for neighbor in neighbors:
            if self in neighbor:
                #TODO : make the ct func random.randint modular so when running we can chose the func
                self.neighbours[(neighbor[1] if neighbor[0] == self else neighbor[0])] = CostTable(self.domain,
                                                                                                   np.random.randint,
                                                                                                   neighbor,
                                                                                                   low= 100,high=200)


    #all dundler functions
    def __str__(self):
        return f"Agent({self.name})"
    #how the class will be shown in debugging or while editing
    def __repr__(self):
        return f"Agent({self.name})"
    def __eq__(self, other):
        if not isinstance(other, Agent):
            return False
        return self.id == other.id
    def __hash__(self):
        return hash(self.id)



class Mailer:
    def send_message(self,agents:List[Agent]):
        pass

