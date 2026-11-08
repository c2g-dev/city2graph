---
seo_title: "Metapath API for Heterogeneous GNNs - City2Graph"
description: "Add metapath-derived edges to heterogeneous spatial graphs with City2Graph, for HAN, HGT and other heterogeneous graph neural network architectures."
keywords: metapath, heterogeneous graph, HAN, HGT, add_metapaths, graph neural networks, edge aggregation, multi-relational graphs, HGNN
---

# Metapath Module

The metapath module provides functions for adding metapath-derived edges to heterogeneous graphs.

A metapath composes typed relations, for example amenity to street segment to amenity, into a direct edge between the end nodes. Heterogeneous graph neural networks such as HAN and HGT use these composite relations. The [metapath tutorial](../examples/add_metapaths.ipynb) builds them for amenities in Soho, London.

::: city2graph.metapath
    options:
      show_root_heading: false
      members:
        - add_metapaths
        - add_metapaths_by_weight
