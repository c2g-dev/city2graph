========
Examples
========

This section provides examples of how to use city2graph in various urban analysis scenarios.

Basic Morphological Graph Creation
----------------------------------

Here's how to create a basic morphological graph from building and road data:

.. code-block:: python

    import geopandas as gpd
    from city2graph.graph import create_morphological_graph
    
    # Load building and road data
    buildings = gpd.read_file("notebooks/liverpool_building.geojson")
    roads = gpd.read_file("notebooks/liverpool_segment.geojson")
    
    # Create a morphological graph connecting buildings and roads
    graph = create_morphological_graph(
        private_gdf=buildings,
        public_gdf=roads,
        private_id_col="fid",
        public_id_col="fid",
        private_node_feature_cols=["height"],
        public_node_feature_cols=["width"]
    )
    
    # Print basic graph statistics
    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")
    
    # Access building and road nodes
    building_nodes = [n for n, data in graph.nodes(data=True) if data['type'] == 'private']
    road_nodes = [n for n, data in graph.nodes(data=True) if data['type'] == 'public']
    print(f"Number of building nodes: {len(building_nodes)}")
    print(f"Number of road nodes: {len(road_nodes)}")

Network Analysis
---------------

Analyzing urban networks with different types of connections:

.. code-block:: python

    import networkx as nx
    from city2graph.morphological_network import (
        create_private_to_private,
        create_private_to_public,
        create_public_to_public
    )
    
    # Create connections between buildings (building-to-building)
    b2b_edges = create_private_to_private(
        buildings,
        private_id_col="fid",
        contiguity="queen",
        distance_threshold=10  # meters
    )
    
    # Create connections between buildings and roads (building-to-road)
    b2r_edges = create_private_to_public(
        buildings,
        roads,
        private_id_col="fid",
        public_id_col="fid",
        distance_threshold=50  # meters
    )
    
    # Create connections between roads (road-to-road)
    r2r_edges = create_public_to_public(
        roads,
        public_id_col="fid"
    )
    
    # Combine all edges to create a comprehensive urban network
    G = nx.Graph()
    G.add_edges_from(b2b_edges)
    G.add_edges_from(b2r_edges)
    G.add_edges_from(r2r_edges)
    
    # Calculate network metrics
    centrality = nx.betweenness_centrality(G)
    
    # Find important nodes in the network
    top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Top 10 most central nodes:")
    for node, score in top_nodes:
        print(f"Node {node}: {score:.4f}")

Working with Jupyter Notebooks
-----------------------------

You can find complete examples in the Jupyter notebooks included in the repository:

- `city2graph_demo.ipynb`: Basic usage demo
- `demo.ipynb`, `demo_2.ipynb`, `demo_3.ipynb`: Additional examples with different scenarios
- `momepy_dev.ipynb`: Advanced examples using momepy integration

PyTorch Geometric Integration
----------------------------

Converting morphological graphs to PyTorch Geometric format for machine learning:

.. code-block:: python

    import torch
    from torch_geometric.data import Data
    
    def networkx_to_pytorch_geometric(graph):
        """Convert a NetworkX graph to PyTorch Geometric format."""
        # Get node features
        node_features = []
        node_types = []
        node_mapping = {}
        
        for i, (node, data) in enumerate(graph.nodes(data=True)):
            node_mapping[node] = i
            
            # Extract features from node data
            features = [
                data.get('height', 0),
                data.get('width', 0),
                data.get('area', 0)
            ]
            
            node_features.append(features)
            node_types.append(1 if data.get('type') == 'private' else 0)
        
        # Create edge index
        edge_index = []
        for source, target in graph.edges():
            edge_index.append([node_mapping[source], node_mapping[target]])
        
        # Convert to PyTorch tensors
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        node_type = torch.tensor(node_types, dtype=torch.long)
        
        # Create PyTorch Geometric data object
        data = Data(x=x, edge_index=edge_index, node_type=node_type)
        return data
    
    # Convert our morphological graph to PyTorch Geometric format
    pyg_data = networkx_to_pytorch_geometric(graph)
    print(f"PyTorch Geometric Data: {pyg_data}")
    
    # Now you can use this with any PyTorch Geometric model
    # For example, with a Graph Neural Network (GNN)

Visualizing Urban Networks
-------------------------

Visualizing the morphological graph to understand urban patterns:

.. code-block:: python

    import matplotlib.pyplot as plt
    import contextily as cx
    
    def plot_morphological_graph(graph, buildings, roads, figsize=(12, 10)):
        """Visualize the morphological graph with buildings and roads."""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot buildings and roads as background
        buildings.plot(ax=ax, color='lightgrey', alpha=0.6)
        roads.plot(ax=ax, color='lightblue', alpha=0.6)
        
        # Plot building-to-building connections
        b2b_edges = [(u, v) for u, v, d in graph.edges(data=True) 
                    if graph.nodes[u]['type'] == 'private' and graph.nodes[v]['type'] == 'private']
        
        # Plot building-to-road connections
        b2r_edges = [(u, v) for u, v, d in graph.edges(data=True) 
                    if (graph.nodes[u]['type'] == 'private' and graph.nodes[v]['type'] == 'public') or
                       (graph.nodes[u]['type'] == 'public' and graph.nodes[v]['type'] == 'private')]
        
        # Plot the network edges by type
        pos = nx.spring_layout(graph)
        nx.draw_networkx_edges(graph, pos, edgelist=b2b_edges, edge_color='blue', alpha=0.5)
        nx.draw_networkx_edges(graph, pos, edgelist=b2r_edges, edge_color='green', alpha=0.5)
        
        # Add basemap
        cx.add_basemap(ax, crs=buildings.crs.to_string())
        
        plt.title("Urban Morphological Graph")
        plt.tight_layout()
        return fig, ax
    
    # Visualize our morphological graph
    fig, ax = plot_morphological_graph(graph, buildings, roads)
    plt.show()