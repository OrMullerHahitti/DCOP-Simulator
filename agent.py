from abc import ABC, abstractmethod
from typing import List, Dict, Callable

import numpy as np

from components import Message, CostTable


class Agent(ABC):
    def __init__(self,id:int ,domain_size:int):
        self.id=id
        self.mailbox : List[Message] = []
        self.domain = domain_size
        self.assignment=np.random.randint(low=0,high=domain_size)
        self.neighbours : Dict[Agent,CostTable] = {}
        self.local_cost = float('inf')



    def receive_message(self, message:Message):
        self.mailbox.append(message)

    ########################################
    def find_assignment(self)-> int:
        temp_cost = float('inf')
        for message in self.mailbox:

            


    @abstractmethod
    def create_new_messages(self) ->List[Message]:
        pass

    @property
    def name(self):
        return f"Agent_{self.id}"


    def set_neighbors(self, neighbors:List[tuple]):
        for neighbor in neighbors:
            if self in neighbor:
                #TODO : make the ct func random.randint modular so when running we can chose the func
                self.neighbours[(neighbor[1] if neighbor[0] == self else neighbor[0])] = CostTable(self.domain,
                                                                                                   np.random.randint,
                                                                                                   neighbor,
                                                                                                   low= 100,high=200)
    def empty_mailbox(self):
        self.mailbox=[]



    #all dunder functions
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


class MGMAgent(Agent):
    def __init__(self,id:int ,name:str,domain_size:int):
        super.__init__(id,name,domain_size)
    def create_new_messages(self) ->List[Message]:
        pass


class Mailer:
    def __init__(self):
        global_cost = float('inf')
        global_assignment :Dict[Agent,int]={}
        mailbox : List[Message]=[]

    def collect_messages(self,agents:List[Agent]):
        for agent in agents:
            self.mailbox.extend(agent.mailbox)
            agent.empty_mailbox()
    def send_messages(self):
        for message in self.mailbox:
            message.receiver.recieve_message(message)



