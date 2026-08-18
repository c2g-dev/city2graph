---
seo_title: "City2Graph: A Python library for Heterogeneous Graph Neural Networks and spatial analysis in urban systems"
description: "Read the abstract and open-access Version of Record of the City2Graph paper published in Computers, Environment and Urban Systems."
hide:
  - navigation
  - toc
---

# City2Graph: A Python library for Heterogeneous Graph Neural Networks and spatial analysis in urban systems { .citation_title }

<p class="paper-authors citation_author">
  Yuta Sato, Elisabetta Pietrostefani, Ron Mahabir, and Daniel Arribas-Bel
</p>

## Open-access Version of Record

Any use of City2Graph in research must cite this paper.

Sato, Y., Pietrostefani, E., Mahabir, R., & Arribas-Bel, D. (2026). City2Graph:
A Python library for Heterogeneous Graph Neural Networks and spatial analysis in
urban systems. *Computers, Environment and Urban Systems*, 130, 102492.
[https://doi.org/10.1016/j.compenvurbsys.2026.102492](https://doi.org/10.1016/j.compenvurbsys.2026.102492)

<div class="paper-actions">
  <a class="md-button md-button--primary" href="assets/papers/sato2026city2graph.pdf" target="_blank" rel="noopener">Open PDF</a>
  <a class="md-button" href="assets/papers/sato2026city2graph.pdf" download>Download PDF</a>
  <a class="md-button" href="https://www.sciencedirect.com/science/article/pii/S0198971526000943" target="_blank" rel="noopener">View on ScienceDirect</a>
</div>

This is the published Version of Record. © 2026 The Author(s). Published by
Elsevier Ltd. under the
[CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/). Use the
[Crossmark record](https://crossmark.crossref.org/dialog?doi=10.1016/j.compenvurbsys.2026.102492&domain=html&date_stamp=2026-07-31)
to check the article's current status and any post-publication updates.

<div class="paper-crossmark-wrapper"><a class="paper-crossmark" href="https://crossmark.crossref.org/dialog?doi=10.1016/j.compenvurbsys.2026.102492&domain=html&date_stamp=2026-07-31" target="_blank" rel="noopener" aria-label="Check the article's current status in Crossmark"><img src="assets/papers/crossmark_logo.svg" alt="Crossmark: Check for updates" width="80" height="80"></a></div>

## Abstract

City2Graph is an open-source Python library that streamlines workflows for heterogeneous Graph Neural Networks (GNNs) in urban systems. Cities are complex systems of diverse spatial relations long modelled as graphs in network science, and recent advances in GNNs have further enabled the identification of non-linear patterns of urban complexity. Unlike homogeneous graphs with a single node and edge type, heterogeneous graphs with multiple types are receiving growing attention to accommodate richer information in GNNs. However, their diffusion remains constrained by fragmented graph construction processes across different data domains, and by the lack of a unified framework for converting constructed graphs into GNN-ready tensors. City2Graph standardises graph construction across domains of morphology, transportation, mobility, and proximity. The library supports conversions between spatial geometries, network topologies, and tensors for spatial analysis with visualisation, network analysis, and GNN training, respectively. City2Graph also supports metapath construction, capturing higher-order connections across node and edge types (e.g., areas linked via multimodal transit). The library’s efficacy was demonstrated through a case study on clustering urban functions in Liverpool, UK, using Graph Autoencoder models with three relation types: spatial contiguity, walk-based accessibility, and multimodal accessibility between census units. Compared with the homogeneous model, heterogeneous models identified spatially coherent clusters that aligned more closely with defined accessibility patterns. City2Graph enables reproducible workflows for heterogeneous GNNs and model interpretation, fostering comprehensive understanding of urban systems across disciplines. The library is released under the BSD 3-Clause License on GitHub.

**Keywords:** Geospatial artificial intelligence (GeoAI); Graph representation
learning; Urban morphology; Multimodal transportation; 15-minute accessibility;
Overture Maps; Open source software.

<div class="paper-pdf-frame">
  <iframe
    src="assets/papers/sato2026city2graph.pdf"
    title="City2Graph open-access paper PDF"
    loading="lazy"
  ></iframe>
</div>

If the embedded viewer is unavailable, use the
[direct PDF link](assets/papers/sato2026city2graph.pdf) or view the
[formal publication on ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0198971526000943).

## BibTeX

```bibtex
@article{sato2026city2graph,
  title = {City2Graph: A Python library for Heterogeneous Graph Neural Networks and spatial analysis in urban systems},
  author = {Sato, Yuta and Pietrostefani, Elisabetta and Mahabir, Ron and Arribas-Bel, Daniel},
  journal = {Computers, Environment and Urban Systems},
  volume = {130},
  pages = {102492},
  year = {2026},
  issn = {0198-9715},
  doi = {10.1016/j.compenvurbsys.2026.102492},
  url = {https://www.sciencedirect.com/science/article/pii/S0198971526000943},
}
```
