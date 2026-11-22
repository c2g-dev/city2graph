"""Shared fixtures for utils tests."""

import networkx as nx
import pytest


@pytest.fixture
def regular_hetero_graph(sample_crs: str) -> nx.Graph:
    """Fixture for regular (non-MultiGraph) heterogeneous graph."""
    graph = nx.Graph()  # Not MultiGraph
    graph.add_node(1, pos=(0, 0), node_type="building")
    graph.add_node(2, pos=(1, 1), node_type="road")
    graph.add_edge(1, 2, edge_type=("building", "connects", "road"))
    graph.graph = {
        "crs": sample_crs,
        "is_hetero": True,
        "node_types": ["building", "road"],
        "edge_types": [("building", "connects", "road")],
    }
    return graph


@pytest.fixture
def simple_nx_graph(sample_crs: str) -> nx.Graph:
    """Fixture for simple NetworkX graph for dual graph testing."""
    graph = nx.Graph()
    graph.add_node(1, pos=(0, 0))
    graph.add_node(2, pos=(1, 1))
    graph.add_edge(1, 2)
    graph.graph = {"crs": sample_crs, "is_hetero": False}
    return graph
