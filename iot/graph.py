"""
Graph generation utilities for distributed constraint optimization.
Contains functions to create constraint graphs for DCOP problems.
"""

from typing import Callable, List, Tuple

import networkx as nx

from iot.core import Agent, Mailer

def generate_constraint_graph(num_agents: int, density: float, domain_size: int,
                             cost_fn: Callable[[int, int], int],
                             agent_class: type[Agent]) -> Tuple[List[Agent], nx.Graph, Mailer]:
    """
    generate a constraint graph for DCOP problems.
    
    Args:
        num_agents: Number of agents in the graph
        density: edge density (probability of edge between any two nodes)
        domain_size: size of the domain for each agent
        cost_fn: function to generate costs for constraints
        agent_class: class to use for creating agents
        
    Returns:
        tuple containing:
        - list of agents
        - NetworkX graph representation
        - Mailer instance for agent communication
    """
    agents = [agent_class(f"A{i}", domain_size) for i in range(num_agents)]
    mailer = Mailer(agents)
    for a in agents:
        a.mailer = mailer

    while True:
        # generate a random graph with the given density
        G = nx.erdos_renyi_graph(num_agents, density)
        if nx.is_connected(G):
            break

    for i, j in G.edges():
        agents[i].connect(agents[j], cost_fn)

    # basic nx graph  for  visualisation
    graph = nx.Graph()
    for i, a in enumerate(agents):
        graph.add_node(a.name, assignment=a.assignment, pos=(i % 5, i // 5))
    for a in agents:
        for n in a.neighbors:
            graph.add_edge(a, n)

    return agents, graph, mailer