---
seo_title: "City2Graph Python API Reference"
description: "City2Graph Python API reference for geospatial graph construction, GTFS and OD networks, proximity graphs, metapaths, and NetworkX or PyG conversion."
---

# City2Graph Python API reference

Use this reference to find the public `city2graph` functions for constructing,
combining, and converting geospatial graphs. Public functions are also exported
from the top-level package, so the examples use `import city2graph as c2g`.

## Find an API by task

| I want to... | Start with | Module |
| --- | --- | --- |
| Convert GeoDataFrames, NetworkX, or PyTorch Geometric graphs | `gdf_to_pyg`, `pyg_to_gdf`, `pyg_to_nx`, `nx_to_pyg` | [Graph](graph.md) |
| Load Overture Maps data or place boundaries | `load_overture_data`, `get_boundaries` | [Data](data.md) |
| Build a building–street morphology graph | `morphological_graph` | [Morphology](morphology.md) |
| Load GTFS or GBFS data into DuckDB | `load_gtfs`, `load_gbfs` | [Transportation](transportation.md) |
| Convert GTFS into a public transport graph | `get_od_pairs`, `travel_summary_graph` | [Transportation](transportation.md) |
| Convert an OD matrix into a weighted spatial graph | `od_matrix_to_graph` | [Mobility](mobility.md) |
| Build proximity, contiguity, bridge, or containment relations | `knn_graph`, `waxman_graph`, `contiguity_graph`, `bridge_nodes`, `group_nodes` | [Proximity](proximity.md) |
| Compose typed relations for a heterogeneous GNN | `add_metapaths`, `add_metapaths_by_weight` | [Metapath](metapath.md) |

## Modules

| Module | Scope |
| --- | --- |
| [Graph](graph.md) | Conversion between GeoDataFrames, NetworkX, and PyTorch Geometric |
| [Data](data.md) | Overture Maps downloads, place boundaries, and geospatial preprocessing |
| [Morphology](morphology.md) | Buildings, tessellations, street segments, and urban form relations |
| [Transportation](transportation.md) | GTFS and GBFS loading, OD pairs, and travel-summary graphs |
| [Mobility](mobility.md) | OD edge lists and adjacency matrices |
| [Proximity](proximity.md) | Proximity, contiguity, bridge, grouping, and directed graph builders |
| [Metapath](metapath.md) | Metapath-derived relations in heterogeneous graphs |
| [Utils](utils.md) | Graph conversion, topology, spatial, and validation utilities |
