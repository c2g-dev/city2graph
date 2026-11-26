---
seo_title: "Utils API: Network Costs, Isochrones, Tessellation, rustworkx - City2Graph"
description: "Utility functions for GeoDataFrame, NetworkX and rustworkx conversion, graph validation, shortest-path OD costs, tessellation, isochrone creation and plotting in City2Graph."
keywords: gdf_to_nx, nx_to_gdf, rustworkx, network_costs, network_cost_matrix, tessellation, isochrone, dual_graph, plot_graph, graph validation, canonicalize_edges, symmetrize_edges
---

# Utils Module

The utils module provides core utility functions for graph conversion, topology operations, network distance and reachability analysis, tessellation, and visualization.

It converts between GeoDataFrames, NetworkX, and rustworkx, validates graph inputs, computes shortest-path origin–destination costs and isochrones for accessibility analysis, creates tessellations, and plots graphs.

For buffer isochrones, see [Filling holes in buffer isochrones](../examples/isochrone_holes.md)
for the optional `fill_holes` setting, a before-and-after figure, and clipping guidance.

## Network costs

Use a projected graph and points in the same CRS. The `cost` column uses the units of the selected edge weight, or planar edge length when `weight` is omitted.

### OSMnx driving times

Given a projected OSMnx graph with edge travel times in seconds, find up to three clinics within 20 minutes of each home. One-way streets and parallel edges retain their NetworkX direction and cost.

```python
import city2graph as c2g

times = c2g.network_costs(homes, clinics, G, weight="travel_time", k=3, threshold=1200)
```

### Overture road accessibility

Given Overture road nodes and directed road arcs with a travel time in seconds, find the nearest facility along projected road edges and combine network time with straight-line access and egress distance (in CRS units).

```python
nearest = c2g.network_costs(
    grid_points,
    facilities,
    (nodes, edges),
    weight="travel_time_sec",
    directed=True,
    k=1,
    connect_to="edge",
    max_connection_distance=1000,
    include_connection_distance=True,
)
nearest["seconds_with_access"] = nearest["cost"] + nearest["connection_distance"] / (4.5 / 3.6)
```

### Streets and GTFS

Combine a street layer, a GTFS `travel_summary_graph()`, and connector edges from `bridge_nodes()` into typed node and edge dictionaries. Restrict point connections to streets while allowing paths to cross all layers.

```python
catchments = c2g.network_cost_matrix(
    homes,
    jobs,
    (node_layers, edge_layers),
    weight="travel_time_sec",
    directed=True,
    connect_node_types="street",
    threshold=2400,
)
pairs = catchments.tocoo()
```

::: city2graph.utils
    options:
      show_root_heading: false
      members:
        - gdf_to_nx
        - nx_to_gdf
        - nx_to_rx
        - rx_to_nx
        - validate_gdf
        - validate_nx
        - canonicalize_edges
        - symmetrize_edges
        - dual_graph
        - network_costs
        - network_cost_matrix
        - filter_graph_by_distance
        - create_isochrone
        - create_tessellation
        - plot_graph
        - clip_graph
        - remove_isolated_components
