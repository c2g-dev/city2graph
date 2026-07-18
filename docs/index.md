---
seo_title: "City2Graph: Geospatial Graphs for Network Analysis and GNNs"
description: "Build spatial and heterogeneous graphs from buildings, streets, GTFS feeds, OD matrices, and POIs for NetworkX analysis and PyTorch Geometric GNNs."
hide:
  - navigation
  - toc
---

# City2Graph: Geospatial Graphs for Network Analysis and GNNs

**City2Graph** is a Python library that turns geospatial data into analysis-ready
graphs. Build networks from buildings, streets, public transport feeds,
origin–destination matrices, zones, and points of interest; analyse them with
[NetworkX](https://networkx.org/) or convert them to
[PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) for
Graph Neural Networks (GNNs).

<p align="center">
  <img src="assets/logos/social_preview_city2graph.png" alt="City2Graph logo" class="desktop-limit-width">
</p>
[![GitHub Stars](https://img.shields.io/github/stars/c2g-dev/city2graph)](https://github.com/c2g-dev/city2graph)
[![PyPI version](https://badge.fury.io/py/city2graph.svg)](https://badge.fury.io/py/city2graph)
[![conda-forge Version](https://anaconda.org/conda-forge/city2graph/badges/version.svg)](https://anaconda.org/conda-forge/city2graph)
[![PyPI Downloads](https://static.pepy.tech/badge/city2graph)](https://pepy.tech/projects/city2graph)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15858845.svg)](https://doi.org/10.5281/zenodo.15858845)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/c2g-dev/city2graph/blob/main/LICENSE)
[![Platform](https://anaconda.org/conda-forge/city2graph/badges/platforms.svg)](https://anaconda.org/conda-forge/city2graph)
[![codecov](https://codecov.io/gh/c2g-dev/city2graph/graph/badge.svg?token=2R449G75Z0)](https://codecov.io/gh/c2g-dev/city2graph)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

City2Graph provides one interface across
[GeoPandas](https://geopandas.org/), NetworkX, PyTorch Geometric, and
[rustworkx](https://www.rustworkx.org/). It preserves geospatial geometries and
attributes while converting between graph representations, so the same graph
can support mapping, spatial network analysis, and machine-learning workflows.

<p align="center">
  <img src="assets/figures/scope.png" alt="City2Graph workflow from geospatial data to NetworkX and PyTorch Geometric graphs" class="desktop-limit-width">
</p>

## Choose a geospatial graph workflow

| Input or task | Graph produced | Main API | Typical use | Guide |
| --- | --- | --- | --- | --- |
| Buildings, streets, and tessellations | Heterogeneous urban morphology graph | [`morphological_graph`](api/morphology.md) | Urban form, walkability, GNN embeddings | [Morphology tutorial](examples/morphological_graph_from_overturemaps.ipynb) |
| GTFS public transport feed | Stop-to-stop travel-time graph | [`travel_summary_graph`](api/transportation.md) | Accessibility, centrality, multimodal routing | [GTFS tutorial](examples/gtfs.ipynb) |
| GBFS shared-mobility JSON feeds | DuckDB tables with station or vehicle point geometry | [`load_gbfs`](api/transportation.md) | Bike-share and shared-mobility preprocessing | [Transportation API](api/transportation.md) |
| OD matrix or flow edge list | Weighted mobility graph | [`od_matrix_to_graph`](api/mobility.md) | Migration, commuting, bike-sharing flows | [OD matrix tutorial](examples/generating_graphs_from_od_matrix.ipynb) |
| Points, polygons, or graph layers | Proximity, contiguity, bridge, or containment graph | [`knn_graph`](api/proximity.md), [`contiguity_graph`](api/proximity.md) | POI access, spatial interaction, zonal adjacency | [Proximity tutorial](examples/generating_graphs_by_proximity.ipynb) |
| Typed relations in a heterogeneous graph | Metapath-derived edges | [`add_metapaths`](api/metapath.md) | Heterogeneous GNNs and composite relations | [Metapath tutorial](examples/add_metapaths.ipynb) |
| GeoDataFrames, NetworkX, PyG, or rustworkx | Converted graph representation | [Graph conversion API](api/graph.md) | Spatial analysis, fast graph algorithms, GNN training | [API reference](api/index.md) |

## Why City2Graph?

- **Geospatial inputs stay geospatial.** Nodes and edges retain geometries,
  coordinate reference systems, and attributes in GeoDataFrames.
- **Heterogeneous graphs are first-class.** Multiple node and relation types can
  represent buildings, streets, transit stops, zones, and amenities together.
- **Conversions work both ways.** Move between GeoDataFrames, NetworkX,
  PyTorch Geometric `Data`/`HeteroData`, and rustworkx without rebuilding the
  graph for every analysis tool.
- **Open urban data is supported.** Load or process Overture Maps,
  OpenStreetMap-derived data, GTFS, GBFS, OD matrices, and common GIS layers.

## Quickstart

```bash
pip install city2graph
```

This installs the graph construction and spatial network analysis features.
Install `city2graph[cpu]` or a CUDA extra when you also need PyTorch Geometric.
See the [installation guide](installation.md) for a package comparison,
supported Python and CUDA versions, and conda-forge instructions.

=== "Morphology"

    ```python
    import city2graph as c2g

    # Buildings + street segments -> heterogeneous morphological graph
    nodes, edges = c2g.morphological_graph(
        buildings_gdf, segments_gdf, center_point, distance=500
    )
    ```

    ![A morphological graph of 500m walking distance in Liverpool](assets/figures/morph_net_overview.png){ .quickstart-figure }

=== "Transportation"

    ```python
    import city2graph as c2g

    gtfs = c2g.load_gtfs("itm_london_gtfs.zip")

    # Stop-to-stop travel-time graph from a GTFS feed
    nodes, edges = c2g.travel_summary_graph(
        gtfs, calendar_start="20250601", calendar_end="20250601"
    )
    ```

    ![A bus transportation graph in London](assets/figures/trav_sum_network_overview.png){ .quickstart-figure }

=== "Mobility"

    ```python
    import city2graph as c2g

    # OD matrix + zone geometries -> weighted spatial graph
    nodes, edges = c2g.od_matrix_to_graph(
        od_df, zones_gdf,
        source_col="origin", target_col="destination",
        weight_cols=["flow"], zone_id_col="zone_id",
    )
    ```

    ![An OD matrix graph showing migration flows and degree centrality in England and Wales](assets/figures/od_matrix_to_graph_uk.png){ .quickstart-figure }

=== "Proximity"

    ```python
    import city2graph as c2g

    # Proximity graphs over points of interest
    knn_nodes, knn_edges = c2g.knn_graph(poi_gdf, k=5)
    wax_nodes, wax_edges = c2g.waxman_graph(poi_gdf, r0=100, beta=0.5)

    # Queen contiguity between polygonal zones
    w_nodes, w_edges = c2g.contiguity_graph(wards_gdf, contiguity="queen")
    ```

    ![Waxman graph of points of interest in Liverpool](assets/figures/waxman_graph.png){ .quickstart-figure }

=== "Metapath"

    ```python
    import city2graph as c2g

    # Compose relations: amenity -> segment -> segment -> amenity
    metapaths = [[("amenity", "is_nearby", "segment"),
                  ("segment", "connects", "segment"),
                  ("segment", "is_nearby", "amenity")]]

    nodes, edges = c2g.add_metapaths(
        (nodes, edges), metapaths, edge_attr="distance_m", edge_attr_agg="sum"
    )
    ```

    ![Animation showing metapath connections between amenities through street segments in Soho, London](assets/figures/metapath.gif){ .quickstart-figure }

=== "Conversions"

    ```python
    import city2graph as c2g

    # GeoPandas -> NetworkX
    G = c2g.gdf_to_nx(nodes, edges)

    # GeoPandas -> PyTorch Geometric
    data = c2g.gdf_to_pyg(nodes, edges)

    # PyTorch Geometric -> GeoPandas
    nodes, edges = c2g.pyg_to_gdf(data)

    # PyTorch Geometric -> NetworkX
    G = c2g.pyg_to_nx(data)

    # NetworkX -> rustworkx
    G_rx = c2g.nx_to_rx(G_nx)

    # rustworkx -> NetworkX
    G_nx = c2g.rx_to_nx(G_rx)
    ```

## Examples

### Applied projects

<div class="grid cards examples-gallery" markdown>

- ![Travel-time network over Liverpool output areas from the case study](assets/examples/case_study.jpg){ .card-img }

    **[Liverpool Case Study](https://github.com/c2g-dev/city2graph-case-study)**

    ---

    A reproducible research pipeline for Liverpool: open data become heterogeneous graphs, graph autoencoders learn embeddings, and clusters characterise urban structure.

- ![city2graph workshop: streets network, walkability, and clustering with GNNs](assets/examples/workshop.jpg){ .card-img }

    **[Workshop: From Geospatial Data to GNNs](https://github.com/c2g-dev/city2graph-workshop)**

    ---

    *GeoAI in Practice*, a hands-on FOSS4G 2026 workshop: construct spatial networks from open data, then build a graph autoencoder pipeline for spatial clustering.

</div>

### Tutorials

<div class="grid cards examples-gallery" markdown>

- ![Walkable street networks of eight city centres extracted from Overture Maps](assets/examples/overture_osmnx.jpg){ .card-img }

    **[How to Use Overture Maps Like OSMnx](https://medium.com/@yuta.sato.now/how-to-use-overture-maps-like-osmnx-by-city2graph-7e01d38f9f61)**

    ---

    Bring the OSMnx-like experience to Overture Maps: fetch buildings, streets, and POIs for any place and turn them into analysis-ready graphs.

- ![Morphological graph of Liverpool: buildings, tessellation cells, and street segments](assets/examples/morphology.jpg){ .card-img }

    **[Morphological Graphs from Overture Maps & OpenStreetMap](examples/morphological_graph_from_overturemaps.ipynb)**

    ---

    Tessellate Liverpool's urban fabric, link it to the street network, and export a heterogeneous graph to NetworkX and PyTorch Geometric.

- ![Metapath edges (cyan) linking amenities across the dual street graph of Soho, London](assets/examples/metapaths.jpg){ .card-img }

    **[Metapath Construction for Heterogeneous GNNs](examples/add_metapaths.ipynb)**

    ---

    Materialise metapath edges between amenities reachable within a few street hops — the composite relations used by heterogeneous GNNs.

- ![Betweenness centrality of every transit stop in London on a dark basemap](assets/examples/gtfs.jpg){ .card-img }

    **[GTFS to Public Transit Graphs](examples/gtfs.ipynb)**

    ---

    Convert a raw GTFS feed for London into a travel-time graph, rank stops by betweenness centrality, and draw walk-plus-transit isochrones.

- ![Migration flows between England and Wales MSOAs drawn as a white network on black](assets/examples/od_matrix.jpg){ .card-img }

    **[OD Matrices to Mobility Graphs](examples/generating_graphs_from_od_matrix.ipynb)**

    ---

    Turn origin–destination data into spatial graphs, up to the 2021 census migration flows between all MSOAs of England and Wales.

- ![Waxman random geometric graph over points of interest in Tokyo](assets/examples/proximity.jpg){ .card-img }

    **[Spatial Proximity Graphs](examples/generating_graphs_by_proximity.ipynb)**

    ---

    Generate KNN, Delaunay, Gilbert, and Waxman graphs over Tokyo POIs under Euclidean, Manhattan, and network distances.

</div>

[Browse all examples →](examples/index.md)

## Frequently asked questions

### What data can City2Graph convert into graphs?

City2Graph supports buildings, street segments, tessellations, Overture Maps
features, GTFS and GBFS feeds, origin–destination matrices, points of interest,
polygonal zones, and existing GeoDataFrame or NetworkX graphs. The
[workflow table](#choose-a-geospatial-graph-workflow) links each input to its
builder and tutorial.

### Is PyTorch required?

No. The core installation builds and analyses geospatial graphs with
GeoPandas, NetworkX, and rustworkx without installing PyTorch. PyTorch and
PyTorch Geometric are optional dependencies for GNN-ready `Data` and
`HeteroData` tensors.

### Can graphs be converted between GeoPandas, NetworkX, and PyTorch Geometric?

Yes. City2Graph supports round-trip conversion between GeoDataFrames, NetworkX,
and PyTorch Geometric, including heterogeneous graphs. See the
[graph conversion API](api/graph.md) for the supported functions.

### How does City2Graph relate to OSMnx?

OSMnx is a focused toolkit for downloading and analysing OpenStreetMap street
networks. City2Graph can use street data from OSMnx and combines it with
buildings, Overture Maps, transit, mobility, proximity, and heterogeneous graph
workflows. The two libraries are complementary.

### How should City2Graph be cited?

Use the project DOI when citing City2Graph in research:

```bibtex
@software{sato2025city2graph,
  title = {City2Graph: Transform geospatial relations into graphs for spatial network analysis and Graph Neural Networks},
  author = {Sato, Yuta},
  year = {2025},
  url = {https://github.com/c2g-dev/city2graph},
  doi = {10.5281/zenodo.15858845},
}
```

<p align="center">
  <a href="https://www.liverpool.ac.uk/geographic-data-science/">
    <img src="assets/logos/gdsl.png" alt="GeoGraphic Data Science Lab" class="footer-logo">
  </a>
</p>
