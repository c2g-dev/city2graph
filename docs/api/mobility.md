---
seo_title: "OD Matrix to Graph API for Mobility Flows - City2Graph"
description: "Convert origin-destination edge lists and adjacency matrices into weighted spatial graphs for migration, commuting and bike-share flow analysis."
keywords: OD matrix, origin-destination, mobility flows, migration, commuting, human mobility, od_matrix_to_graph, flow networks, spatial interaction
---

# Mobility Module

The mobility module provides functions for processing Origin-Destination (OD) matrices and creating mobility networks for urban flow analysis.

`od_matrix_to_graph` accepts an edge list or an adjacency matrix together with zone geometries and returns a weighted spatial graph of migration, commuting, or bike-sharing flows. The [OD matrix tutorial](../examples/generating_graphs_from_od_matrix.ipynb) builds one from 2021 census migration data for England and Wales.

## Travel time and accessibility

Use a projected graph and points in the same CRS. The `cost` column has the units
of the selected edge weight. Omitting the weight uses planar edge length.

### OSMnx driving times

Given a projected OSMnx graph with edge travel times in seconds, find up to three
clinics within 20 minutes of each home. One-way streets and parallel edges keep
their NetworkX direction and cost.

```python
import city2graph as c2g

times = c2g.network_costs(homes, clinics, G, weight="travel_time", k=3, threshold=1200)
```

### Overture road accessibility

Given Overture road nodes and directed road arcs with a travel time in seconds,
prepare the network once and reuse the same population grid connections for each
facility category. Coordinates in this example are in metres.

```python
roads = c2g.PreparedNetwork(
    (nodes, edges), directed=True, weight="travel_time_sec", connect_to="edge"
)
population = roads.connect(grid_points, max_connection_distance=1000)
for category, facilities in facilities_gdf.groupby("category"):
    nearest = c2g.network_costs(
        population, facilities, roads, k=1, max_connection_distance=1000,
        include_connection_distance=True,
    )
    nearest["seconds_with_access"] = nearest.cost + nearest.connection_distance / (4.5 / 3.6)
```

The access distance includes both endpoint connections. It is excluded from the
network cost, the threshold and the nearest-destination ranking. A separate
destination set and `k=1` use one reverse multi-source search. Within-set nearest-neighbour queries track two distinct destinations per
vertex to exclude self-pairs. Other queries search in batches from the side with fewer unique connected vertices.

### Streets and GTFS

Combine a street layer, a GTFS `travel_summary_graph()` and connector edges from
`bridge_nodes()` into typed node and edge dictionaries. Give every traversable
arc (including walking and transfers) a non-negative `travel_time_sec`, and
include reverse arcs for two-way walking. Restrict point connections to streets
while allowing paths to cross all layers.

```python
catchments = c2g.network_cost_matrix(
    homes, jobs, (node_layers, edge_layers), weight="travel_time_sec",
    directed=True, connect_node_types="street", threshold=2400,
)
pairs = catchments.tocoo()
# pairs.row and pairs.col are input positions. pairs.data includes reachable zeros.
```

The long table keeps input index labels, including duplicates. Origins without
a qualifying destination contribute no rows. In the matrix they retain their
row position. Omitting destinations compares origins to each other, excluding
each row's own position when requesting its nearest neighbours.

::: city2graph.mobility
    options:
      show_root_heading: false
      members:
        - od_matrix_to_graph
        - network_costs
        - network_cost_matrix
        - PreparedNetwork
