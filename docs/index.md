---
description: City2Graph is a Python library for converting geospatial datasets into graphs for Graph Neural Networks (GNN) and spatial analysis.
keywords: GeoAI, Graph Neural Networks, GNN, PyTorch Geometric, Geospatial Analysis, Urban Analytics, Spatial Data Science, Urban Mobility, Transportation Networks, Geospatial Foundation Models, Digital Twin, Urban Informatics, Geographic Data Science, Graph Representation Learning, Urban Planning, Urban Morphology, Accessibility Analysis
hide:
  navigation
---

# City2Graph

<p align="center">
  <img src="assets/logos/social_preview_city2graph.png" alt="City2Graph logo">
</p>

**City2Graph** is a Python library for converting geospatial datasets into graphs for GNN with integrated interface of [GeoPandas](https://geopandas.org/), [NetworkX](https://networkx.org/), and [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) across multiple domains (e.g. streets, transportations, OD matrices, POI proximities, etc.). It enables researchers and practitioners to seamlessly develop advanced GeoAI and geographic data science applications.

<p align="center">
  <img src="assets/figures/scope.png" alt="Overview scope of City2Graph" class="desktop-limit-width">
</p>

## Features

- Construct graphs from morphological datasets (e.g. buildings, streets, and land use from OpenStreetMap (OSM), Overture Maps, and others)
- Construct graphs from transportation datasets (e.g. public transport of buses, trams, and trains from GTFS) with DuckDB-backed loading and aggregation
- Construct graphs from contiguity datasets (e.g. land use, land cover, and administrative boundaries)
- Construct graphs from mobility datasets (e.g. bike-sharing, migration, and pedestrian flows)
- Convert geospatial data (GeoPandas / NetworkX) into tensors (PyTorch Geometric's Data and HeteroData) for graph representation learning, such as Graph Neural Networks (GNNs)
- Build multi-center accessibility catchments and layered isochrones from network graphs

**City2Graph** empowers researchers and practitioners in GeoAI, Urban Analytics, and Spatial Data Science to build advanced applications. A key distinction of this library is its ability to model complex urban systems by handling multiple geospatial relations as **heterogeneous graphs**. It bridges the gap between traditional GIS and modern Graph Neural Networks (GNNs) for a variety of applications. By supporting standard libraries like PyTorch Geometric, it enables seamless integration into deep learning workflows for Graph Representation Learning. With its versatile graph construction interface, this library can also be used for network analysis of urban systems from multiple geospatial relations, such as multi-modal accessibility (e.g. isochrone with street networks + public transport networks).

[![GitHub Stars](https://img.shields.io/github/stars/c2g-dev/city2graph)](https://github.com/c2g-dev/city2graph)
[![PyPI version](https://badge.fury.io/py/city2graph.svg)](https://badge.fury.io/py/city2graph)
[![conda-forge Version](https://anaconda.org/conda-forge/city2graph/badges/version.svg)](https://anaconda.org/conda-forge/city2graph)
[![PyPI Downloads](https://static.pepy.tech/badge/city2graph)](https://pepy.tech/projects/city2graph)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15858845.svg)](https://doi.org/10.5281/zenodo.15858845)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/c2g-dev/city2graph/blob/main/LICENSE)
[![Platform](https://anaconda.org/conda-forge/city2graph/badges/platforms.svg)](https://anaconda.org/conda-forge/city2graph)
[![codecov](https://codecov.io/gh/c2g-dev/city2graph/graph/badge.svg?token=2R449G75Z0)](https://codecov.io/gh/c2g-dev/city2graph)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Quickstart

Install City2Graph using pip (recommended):

```bash
pip install city2graph
```

For graph neural network functionality (PyTorch, PyTorch Geometric):

```bash
pip install "city2graph[cpu]"  # CPU only
pip install "city2graph[cu130]"  # For CUDA 13.0 (GPU)
```

Supported CUDA options: cu126, cu128, cu130

The CPU, cu126, and cu130 options use PyTorch 2.12 or newer. The cu128 option
uses PyTorch 2.11 because CUDA 12.8 wheels are no longer published for
PyTorch 2.12.

Or install with conda (core functionality only):

```bash
conda install -c conda-forge city2graph
```

To add PyTorch and PyTorch Geometric (conda):

```bash
conda install -c conda-forge pytorch pytorch_geometric
```

For CUDA-enabled PyTorch (conda):

```bash
conda install -c conda-forge pytorch=2.12.0=*cuda130*
conda install -c conda-forge pytorch_geometric
```

!!! warning
    As conda is not officially supported by PyTorch and PyTorch Geometric anymore, pip is recommended for full PyTorch support. See [Installation](installation.md) for details.

## Examples

```python
import city2graph as c2g
import geopandas as gpd
import networkx as nx
import pandas as pd
from pathlib import Path
import rustworkx as rx
```

**Metapath**

```python
# Define metapath: amenity -> segment -> segment -> amenity (3-hop in this case)
metapaths = [[("amenity", "is_nearby", "segment"),
              ("segment", "connects", "segment"),
              ("segment", "connects", "segment"),
              ("segment", "is_nearby", "amenity")]]

# Add metapath-derived edges to connect amenities through street networks
nodes_with_metapaths, edges_with_metapaths = c2g.add_metapaths(
    (combined_nodes, combined_edges),
    metapaths,
    edge_attr="distance_m",
    edge_attr_agg="sum"
)
```

<p align="center">
  <img src="assets/figures/metapath.gif" alt="Animation showing metapath connections between amenities through street segments in Soho, London" class="desktop-limit-width">
</p>

For details, see [Examples](examples/add_metapaths.ipynb)

**Morphology**

```python
morphological_nodes, morphological_edges = c2g.morphological_graph(
   buildings_gdf,
   segments_gdf,
   center_point,
   distance=500
 )
```

<p align="center">
  <img src="assets/figures/morph_net_overview.png" alt="A morphological graph of 500m walking distance in Liverpool" class="desktop-limit-width">
</p>

For details, see [Examples](examples/morphological_graph_from_overturemaps.ipynb)

**Transportation**

```python
sample_gtfs_path = Path("./itm_london_gtfs.zip")
gtfs_con = c2g.load_gtfs(sample_gtfs_path)

travel_summary_nodes, travel_summary_edges = c2g.travel_summary_graph(
   gtfs_con,
   calendar_start="20250601",
   calendar_end="20250601",
)

travel_summary_graph = c2g.gdf_to_nx(travel_summary_nodes, travel_summary_edges)
travel_rx_graph = c2g.nx_to_rx(travel_summary_graph)

import rustworkx as rx
betweenness_centrality = rx.betweenness_centrality(travel_rx_graph)

# Set the betweenness centrality as a node attribute
nx.set_node_attributes(
  travel_summary_graph,
  betweenness_centrality,
  "betweenness_centrality",
)
```

<p align="center">
  <img src="assets/figures/trav_sum_network_overview.png" alt="A bus transportation graph in London" class="desktop-limit-width">
</p>

For details, see [Examples](examples/gtfs.ipynb)

**Mobility**

```python
# Load zones (e.g., MSOA boundaries) and OD matrix data
od_data = pd.read_csv("od_matrix.csv")
zones_gdf = gpd.read_file("zones.gpkg")

# Convert OD matrix to graph
od_nodes, od_edges = c2g.od_matrix_to_graph(
    od_data,
    zones_gdf,
    source_col="origin",
    target_col="destination",
    weight_cols=["flow"],
    zone_id_col="zone_id",
    directed=False
)
```

<p align="center">
  <img src="assets/figures/od_matrix_to_graph_uk.png" alt="An OD matrix graph showing migration flows and degree centrality in England and Wales" class="desktop-limit-width">
</p>

For details, see [Examples](examples/generating_graphs_from_od_matrix.ipynb)

**Proximity**

```python
fixed_radius_graph = c2g.fixed_radius_graph(poi_gdf, radius=100)
```

<p align="center">
  <video class="desktop-limit-width" controls>
    <source src="assets/videos/gilbert_graph.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</p>

```python
wax_l1_nodes, wax_l1_edges = c2g.waxman_graph(poi_gdf,
                                 distance_metric="manhattan",
                                 r0=100,
                                 beta=0.5)

wax_l2_nodes, wax_l2_edges = c2g.waxman_graph(poi_gdf,
                                 distance_metric="euclidean",
                                 r0=100,
                                 beta=0.5)

wax_net_nodes, wax_net_edges = c2g.waxman_graph(poi_gdf,
                                 distance_metric="network",
                                 r0=100,
                                 beta=0.5,
                                 network_gdf=segments_gdf.to_crs(epsg=6677))
```

<p align="center">
  <img src="assets/figures/waxman_graph.png" alt="Waxman graph of points of interest in Liverpool" class="desktop-limit-width">
</p>

```python
nodes_dict = {
   "restaurants": poi_gdf,
   "hospitals": hospital_gdf,
   "commercial": commercial_gdf
}

# Generate proximity edges between layers using KNN method
proximity_nodes, proximity_edges = c2g.bridge_nodes(
   nodes_dict,
   proximity_method="knn",
   k=5,
   distance_metric="euclidean"
)
```

<p align="center">
  <img src="assets/figures/bridge_nodes.png" alt="Bridge nodes connecting different layers of POIs" class="desktop-limit-width">
</p>

```python
# Build a contiguity graph (Queen or Rook) from polygonal zones
wn_q_nodes, wn_q_edges = c2g.contiguity_graph(
    wards_gdf,
    contiguity="queen",        # or "rook"
    distance_metric="euclidean" # or "manhattan", "network"
)
```

```python
# Link point features (e.g., POIs, stops) to containing polygons (e.g., wards)
nodes_dict, edges_dict = c2g.group_nodes(
    polygons_gdf=wards_gdf,
    points_gdf=poi_gdf,
    predicate="covered_by"      # include boundary points; alternatives: "within", "contains"
)
```

```python
# Combine contiguity edges and grouped edges into a single heterogeneous graph
combined_nodes = {
    "wards": wn_q_nodes,
    "poi": nodes_dict["poi"]
}

combined_edges = {
    ("wards", "is_contiguous_with", "wards"): wn_q_edges[("wards", "is_contiguous_with", "wards")],
    ("wards", "covers", "poi"): edges_dict[("wards", "covers", "poi")]
}

# Convert to PyG HeteroData
hetero_graph = c2g.gdf_to_pyg(combined_nodes, combined_edges)
```

<p align="center">
  <img src="assets/figures/contiguity_graph.png" alt="Contiguity graph of wards (MSOA) with grouped connections of bus stations in London" class="desktop-limit-width">
</p>

For details, see [Examples](examples/generating_graphs_by_proximity.ipynb)

## Citation

If you use City2Graph in your research, please cite it as follows:

```bibtex
@software{sato2025city2graph,
  title = {City2Graph: Transform geospatial relations into graphs for spatial network analysis and Graph Neural Networks},
  author = {Sato, Yuta},
  year = {2025},
  url = {https://github.com/c2g-dev/city2graph},
  doi = {10.5281/zenodo.15858845},
}
```

You can also find the citation information in the [CITATION.cff](https://github.com/c2g-dev/city2graph/blob/main/CITATION.cff) file in the repository, which follows the Citation File Format standard.

## Documentation

- [Installation](installation.md)
- [Examples](examples/index.md)
- [API Reference](api/index.md)
- [Contributing](contributing.md)

<p align="center">
  <a href="https://www.liverpool.ac.uk/geographic-data-science/">
    <img src="assets/logos/gdsl.png" alt="GeoGraphic Data Science Lab">
  </a>
</p>
