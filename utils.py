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

def create_graph(agents_num:int, domain_size:int, probability,  ct_function:Callable, **kwargs)->List[Union[Agent,CostTable]]:
    """
    Create a graph with the specified number of agents and domain size.

    Args:
        agents_num (int): The number of agents in the graph.
        domain_size (int): The size of the domain for each agent.
        ct_function (Callable): The function to create the cost table.
        **kwargs: Additional parameters for the cost table creation function.

    Returns:
        list: A list of agents with their respective cost tables.
    """
    agents = create_agents(agents_num, domain_size)
    neighbors = create_neighbors(agents, probability) #returns a list of tuples
    for agent in agents:
        agent.set_neighbors(neighbors)

