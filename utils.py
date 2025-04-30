import random
from itertools import combinations
from typing import List, Optional, Callable
import networkx as nx
from numpy import ndarray

from agent import Agent  # your existing Agent class

def create_random_connected_graph(
    num_variables: int,
    domain_size: int,
    density: float,
    seed: Optional[int] = None
) -> nx.Graph:
    """
    Create a connected NetworkX graph whose nodes are Agent instances,
    with exactly floor(density * max_edges) edges.

    Args:
        num_variables (int): number of Agents
        domain_size (int): domain size passed for each agent
        density (float): in [0,1], fraction of total possible edges
        seed (Optional[int]): RNG seed for reproducibility

    Returns:
        nx.Graph: connected graph on Agent nodes
    """
    if not (0 <= density <= 1):
        raise ValueError("density must be between 0 and 1")

    n = num_variables
    max_edges = n * (n - 1) // 2
    target_m = int(density * max_edges)
    if target_m < n - 1:
        min_d = (n - 1) / max_edges
        raise ValueError(f"density too low; need at least {min_d:.3f} to connect {n} nodes")

    #create Agent nodes
    agents: List[Agent] = [Agent(i, domain_size) for i in range(n)]

    #start with a random tree to gurantee connectivity
    tree = nx.random_tree(agents, seed=seed)
    G = nx.Graph()
    G.add_nodes_from(agents)
    G.add_edges_from(tree.edges())

    # 3) add extra edges up to target_m - amount of edges
    all_pairs = list(combinations(agents, 2)) #all possible pairs of agents
    non_tree = [edge for edge in all_pairs if not G.has_edge(*edge)] # pairs not in the tree
    rnd = random.Random(seed) # for reproducibility
    extras = rnd.sample(non_tree, max(0,target_m - (n - 1)))
    G.add_edges_from(extras)

    return G

