"""Tests for the public exports of city2graph.utils."""

from __future__ import annotations

import pytest

from city2graph import utils
from city2graph.utils import conversion as conversion_utils
from city2graph.utils import distance as distance_utils
from city2graph.utils import isochrone as isochrone_utils
from city2graph.utils import plotting as plotting_utils
from city2graph.utils import tessellation as tessellation_utils
from city2graph.utils import topology as topology_utils


class TestPublicUtilityExports:
    """Verify compatibility exports and focused public submodules."""

    def test_root_all_matches_public_api(self) -> None:
        """The package-level wildcard API should match the expected utility exports."""
        assert utils.__all__ == [
            "canonicalize_edges",
            "clip_graph",
            "create_isochrone",
            "create_tessellation",
            "dual_graph",
            "filter_graph_by_distance",
            "gdf_to_nx",
            "network_cost_matrix",
            "network_costs",
            "nx_to_gdf",
            "nx_to_rx",
            "plot_graph",
            "remove_isolated_components",
            "rx_to_nx",
            "symmetrize_edges",
            "validate_gdf",
            "validate_nx",
        ]

    @pytest.mark.parametrize(
        ("name", "owner"),
        [
            ("gdf_to_nx", conversion_utils),
            ("nx_to_gdf", conversion_utils),
            ("nx_to_rx", conversion_utils),
            ("rx_to_nx", conversion_utils),
            ("validate_gdf", conversion_utils),
            ("validate_nx", conversion_utils),
            ("canonicalize_edges", topology_utils),
            ("clip_graph", topology_utils),
            ("dual_graph", topology_utils),
            ("remove_isolated_components", topology_utils),
            ("symmetrize_edges", topology_utils),
            ("filter_graph_by_distance", distance_utils),
            ("network_cost_matrix", distance_utils),
            ("network_costs", distance_utils),
            ("create_isochrone", isochrone_utils),
            ("create_tessellation", tessellation_utils),
            ("plot_graph", plotting_utils),
        ],
    )
    def test_root_exports_are_direct_aliases(self, name: str, owner: object) -> None:
        """Root utility exports should directly alias their owning implementation."""
        assert getattr(utils, name) is getattr(owner, name)

    def test_legacy_non_wildcard_exports_remain_available(self) -> None:
        """Previously importable compatibility names should remain accessible."""
        assert utils.NxConverter is conversion_utils.NxConverter
        assert utils.MATPLOTLIB_AVAILABLE is plotting_utils.MATPLOTLIB_AVAILABLE
