import random
from abc import ABC, abstractmethod
from typing import List, Dict, Callable
import matplotlib.pyplot as plt
import networkx as nx

import numpy as np

from components import Message, CostTable
from utils import create_random_connected_graph


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



class AgentGraph():
    def __init__(self,num_variables:int,domain_size:int,density:float,seed:int,ct_creation:Callable,ct_kwargs:Dict):
        self.g = create_random_connected_graph(num_variables,domain_size,density,seed)
        self.domain_size = domain_size
        self.iteration = 0
        self.global_cost = float('inf')
        self.mailer = Mailer()
        self.ct_creation = ct_creation
        self.ct_kwargs = ct_kwargs
        self._set_constraints()
        self.agents = list(self.g.nodes())

    def _set_constraints(self):
        for edge in self.g.edges():
            agent1,agent2=edge
            ct = CostTable((agent1,agent2),self.domain_size,self.ct_creation,**self.ct_kwargs)
            agent1.neighbours[agent2] = ct
            agent2.neighbours[agent1] = ct
    def visualize(self):
        """
        Visualize the graph using NetworkX and Matplotlib.
        """

        G = nx.Graph()
        G.add_nodes_from(self.agents)
        G.add_edges_from(self.g.edges())

        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10)
        plt.show()


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



