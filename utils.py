from typing import Callable, Union, List

import numpy as np

from agent import Agent



def create_table(domain_size: int, ct_function,**kwargs):
    """
    Create a cost table using the specified function and parameters.

    Args:
        domain_size (int): The size of the domain for the cost table.
        ct_function (Callable): The function to create the cost table.
        **kwargs: Additional parameters for the cost table creation function.

    Returns:
        np.ndarray: The created cost table.
    """
    shape = (domain_size,)*2
    return ct_function(shape, **kwargs)
class Graph_Creator:
    def __init__(self):
        self.agents = []
        self.neighbors = []
        self.cost_tables = []
    def _create_agents(self, agents_num:int, domain_size:int)->List[Agent]:
        """
        Create a list of agents with the specified number and domain size.

        Args:
            agents_num (int): The number of agents to create.
            domain_size (int): The size of the domain for each agent.

        Returns:
            list: A list of created agents.
        """
        return [Agent(i, domain_size) for i in range(agents_num)]
    def _create_all_edges(self, agents:List[Agent], probability:float)->List[tuple[Agent, Agent]]:
        """
        Create a list of neighbors for each agent based on the specified probability.

        Args:
            agents (list): A list of agents.
            probability (float): The probability of creating a neighbor
        """
        return
