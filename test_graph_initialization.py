import pytest
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

@pytest.fixture
def random_graph():
    # Create a random graph with weights between 100 and 200
    np.random.seed(42)  # For reproducibility
    G = nx.erdos_renyi_graph(n=30, p=0.5)  # 10 nodes, 50% probability of edge creation
    for (u, v) in G.edges():
        G[u][v]['weight'] = np.random.randint(100, 200)
    return G

def test_graph_initialization(random_graph):
    G = random_graph
    # Assert that all edge weights are within the range 100-200
    for (u, v, data) in G.edges(data=True):
        assert 100 <= data['weight'] < 200

def test_graph_visualization(random_graph):
    G = random_graph
    pos = nx.spring_layout(G)  # Layout for visualization
    weights = nx.get_edge_attributes(G, 'weight')

    # Visualize the graph
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)
    plt.title("Random Graph with Weights")
    plt.show()
