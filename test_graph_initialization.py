import pytest
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import logging
import sys

# Configure logging to stream to stdout
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

@pytest.fixture
def random_graph():
    logger.info("Creating a random graph with weights between 100 and 200.")
    np.random.seed(42)  # For reproducibility
    G = nx.erdos_renyi_graph(n=30, p=0.5)
    for (u, v) in G.edges():
        G[u][v]['weight'] = np.random.randint(100, 200)
    logger.info("Random graph created successfully.")
    return G

def test_graph_initialization(random_graph):
    logger.info("Starting test_graph_initialization")
    G = random_graph
    for (u, v, data) in G.edges(data=True):
        assert 100 <= data['weight'] < 200
    logger.info("All edge weights are within the range 100-200.")

def test_graph_visualization(random_graph):
    logger.info("Starting test_graph_visualization")
    G = random_graph
    pos = nx.spring_layout(G)
    weights = nx.get_edge_attributes(G, 'weight')

    logger.info("Visualizing the graph.")
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)
    plt.title("Random Graph with Weights")
    plt.show()
    logger.info("Graph visualization completed.")
def test_visualization(random_graph):
    logger.info("Starting test_visualization")
    G = random_graph
    G.visualize()
