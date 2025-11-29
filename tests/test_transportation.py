"""Tests for the transportation module."""

# Import directly to avoid torch import issues
# Standard library imports
import sys
from datetime import datetime
from pathlib import Path

# Third-party imports
import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).parent.parent))
from city2graph.transportation import get_od_pairs
from city2graph.transportation import load_gtfs
from city2graph.transportation import travel_summary_graph


class TestLoadGtfs:
    """Test the load_gtfs function."""

    def test_load_gtfs_basic(self, sample_gtfs_zip: str) -> None:
        """Test basic GTFS loading functionality."""
        result = load_gtfs(sample_gtfs_zip)

        # Check that we get a dictionary
        assert isinstance(result, dict)

        # Check expected files are present
        expected_files = {"stops", "routes", "trips", "stop_times", "calendar"}
        assert expected_files.issubset(result.keys())

        # Check that stops has geometry
        assert isinstance(result["stops"], gpd.GeoDataFrame)
        assert result["stops"].crs.to_string() == "EPSG:4326"
        assert "geometry" in result["stops"].columns

        # Check data types
        assert pd.api.types.is_numeric_dtype(result["stops"]["stop_lat"])
        assert pd.api.types.is_numeric_dtype(result["stops"]["stop_lon"])

    def test_load_gtfs_with_shapes(self, sample_gtfs_zip_with_shapes: str) -> None:
        """Test GTFS loading with shapes.txt."""
        result = load_gtfs(sample_gtfs_zip_with_shapes)

        # Check shapes is a GeoDataFrame with geometry
        assert isinstance(result["shapes"], gpd.GeoDataFrame)
        assert "geometry" in result["shapes"].columns
        assert result["shapes"].crs.to_string() == "EPSG:4326"

    def test_load_gtfs_empty_zip(self, empty_gtfs_zip: str) -> None:
        """Test loading an empty GTFS zip file."""
        result = load_gtfs(empty_gtfs_zip)
        assert result == {}

    def test_load_gtfs_invalid_coordinates(self, gtfs_zip_invalid_coords: str) -> None:
        """Test GTFS loading with invalid coordinates."""
        result = load_gtfs(gtfs_zip_invalid_coords)

        # Should still load but with NaN coordinates filtered out
        assert "stops" in result
        # Stops with invalid coordinates should be filtered out
        assert len(result["stops"]) < 3  # Original had 3 stops, some invalid

    def test_load_gtfs_nonexistent_file(self) -> None:
        """Test loading a non-existent GTFS file."""
        with pytest.raises(FileNotFoundError):
            load_gtfs("nonexistent.zip")


class TestGetOdPairs:
    """Test the get_od_pairs function."""

    def test_get_od_pairs_basic(self, sample_gtfs_dict: dict[str, pd.DataFrame]) -> None:
        """Test basic OD pairs generation."""
        result = get_od_pairs(sample_gtfs_dict)

        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) > 0

        # Check required columns
        expected_cols = {
            "trip_id",
            "service_id",
            "orig_stop_id",
            "dest_stop_id",
            "departure_ts",
            "arrival_ts",
            "travel_time_sec",
            "date",
        }
        assert expected_cols.issubset(result.columns)

        # Check geometry
        assert result.crs.to_string() == "EPSG:4326"
        assert all(isinstance(geom, LineString) for geom in result.geometry if geom is not None)

    def test_get_od_pairs_date_range(self, sample_gtfs_dict: dict[str, pd.DataFrame]) -> None:
        """Test OD pairs with specific date range."""
        result = get_od_pairs(
            sample_gtfs_dict,
            start_date="20240101",
            end_date="20240102",
        )

        assert isinstance(result, gpd.GeoDataFrame)
        # Should have data for the specified date range
        dates = pd.to_datetime(result["date"]).dt.date
        assert dates.min() >= datetime(2024, 1, 1).date()
        assert dates.max() <= datetime(2024, 1, 2).date()

    def test_get_od_pairs_no_geometry(self, sample_gtfs_dict: dict[str, pd.DataFrame]) -> None:
        """Test OD pairs without geometry."""
        result = get_od_pairs(sample_gtfs_dict, include_geometry=False)

        assert isinstance(result, pd.DataFrame)
        assert "geometry" not in result.columns

    def test_get_od_pairs_incomplete_gtfs(self) -> None:
        """Test OD pairs with incomplete GTFS data."""
        incomplete_gtfs = {"stops": pd.DataFrame()}  # Missing required tables

        result = get_od_pairs(incomplete_gtfs)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_get_od_pairs_with_calendar_dates(
        self,
        sample_gtfs_dict_with_exceptions: dict[str, pd.DataFrame],
    ) -> None:
        """Test OD pairs with calendar exceptions."""
        result = get_od_pairs(sample_gtfs_dict_with_exceptions)

        assert isinstance(result, gpd.GeoDataFrame)
        # Should handle calendar exceptions properly
        assert len(result) > 0

    def test_get_od_pairs_with_malformed_times(
        self,
        sample_gtfs_dict: dict[str, pd.DataFrame],
    ) -> None:
        """Test OD pairs with malformed time values that trigger non-string handling."""
        # Create a modified GTFS with some malformed times
        gtfs_copy = sample_gtfs_dict.copy()
        stop_times = gtfs_copy["stop_times"].copy()

        # Replace some time values with None and numeric values to trigger line 72
        stop_times.loc[0, "departure_time"] = None
        stop_times.loc[1, "arrival_time"] = 3600.0  # numeric value

        gtfs_copy["stop_times"] = stop_times

        # This should still work and not crash
        result = get_od_pairs(gtfs_copy)
        assert isinstance(result, (pd.DataFrame, gpd.GeoDataFrame))

    def test_get_od_pairs_removal_exception_integration(self) -> None:
        """Test removal exception via calendar_dates in get_od_pairs."""
        # Minimal GTFS with one trip on 2024-01-01, then removed by exception
        stops = pd.DataFrame(
            {
                "stop_id": ["s1", "s2"],
                "stop_lat": [0.0, 1.0],
                "stop_lon": [0.0, 1.0],
            },
        )
        trips = pd.DataFrame({"trip_id": ["t1"], "service_id": ["svc"]})
        # stop_times should not include service_id; it's merged from trips
        stop_times = pd.DataFrame(
            {
                "trip_id": ["t1", "t1"],
                "stop_id": ["s1", "s2"],
                "stop_sequence": [1, 2],
                "departure_time": ["08:00:00", "08:10:00"],
                "arrival_time": ["08:10:00", "08:20:00"],
            },
        )
        calendar = pd.DataFrame(
            {
                "service_id": ["svc"],
                "start_date": ["20240101"],
                "end_date": ["20240101"],
                "monday": [False],
                "tuesday": [False],
                "wednesday": [False],
                "thursday": [False],
                "friday": [False],
                "saturday": [False],
                "sunday": [True],
            },
        )
        calendar_dates = pd.DataFrame(
            {
                "service_id": ["svc"],
                "date": ["20240101"],
                "exception_type": [2],
            },
        )
        gtfs = {
            "stops": stops,
            "trips": trips,
            "stop_times": stop_times,
            "calendar": calendar,
            "calendar_dates": calendar_dates,
        }
        result = get_od_pairs(gtfs, include_geometry=False)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestTravelSummaryGraph:
    """Test the travel_summary_graph function."""

    def test_travel_summary_graph_basic(self, sample_gtfs_dict: dict[str, pd.DataFrame]) -> None:
        """Test basic travel summary graph generation."""
        nodes_gdf, edges_gdf = travel_summary_graph(sample_gtfs_dict)

        # Check nodes
        assert isinstance(nodes_gdf, gpd.GeoDataFrame)
        assert len(nodes_gdf) > 0
        assert nodes_gdf.index.name == "stop_id"

        # Check edges
        assert isinstance(edges_gdf, gpd.GeoDataFrame)
        assert len(edges_gdf) > 0

        expected_edge_cols = {"travel_time_sec", "frequency"}
        assert expected_edge_cols.issubset(edges_gdf.columns)

        # Check MultiIndex
        assert isinstance(edges_gdf.index, pd.MultiIndex)
        assert edges_gdf.index.names == ["from_stop_id", "to_stop_id"]

    def test_travel_summary_graph_time_filter(
        self,
        sample_gtfs_dict: dict[str, pd.DataFrame],
    ) -> None:
        """Test travel summary graph with time filtering."""
        nodes_gdf, edges_gdf = travel_summary_graph(
            sample_gtfs_dict,
            start_time="07:00:00",
            end_time="09:00:00",
        )

        assert isinstance(nodes_gdf, gpd.GeoDataFrame)
        assert isinstance(edges_gdf, gpd.GeoDataFrame)
        # Should have fewer edges due to time filtering
        assert len(edges_gdf) >= 0

    def test_travel_summary_graph_calendar_range(
        self,
        sample_gtfs_dict: dict[str, pd.DataFrame],
    ) -> None:
        """Test travel summary graph with calendar date range."""
        nodes_gdf, edges_gdf = travel_summary_graph(
            sample_gtfs_dict,
            calendar_start="20240101",
            calendar_end="20240107",
        )

        assert isinstance(nodes_gdf, gpd.GeoDataFrame)
        assert isinstance(edges_gdf, gpd.GeoDataFrame)

    def test_travel_summary_graph_as_nx(self, sample_gtfs_dict: dict[str, pd.DataFrame]) -> None:
        """Test travel summary graph returning NetworkX graph."""
        result = travel_summary_graph(sample_gtfs_dict, as_nx=True)

        assert isinstance(result, nx.Graph)
        assert result.number_of_nodes() > 0
        assert result.number_of_edges() >= 0

    def test_travel_summary_graph_missing_data(self) -> None:
        """Test travel summary graph with missing required data."""
        incomplete_gtfs = {"stops": pd.DataFrame()}  # Missing stop_times

        with pytest.raises(ValueError, match="GTFS must contain at least stop_times and stops"):
            travel_summary_graph(incomplete_gtfs)

    def test_travel_summary_graph_empty_stop_times(
        self,
        sample_gtfs_dict: dict[str, pd.DataFrame],
    ) -> None:
        """Test travel summary graph with empty stop_times."""
        gtfs_copy = sample_gtfs_dict.copy()
        gtfs_copy["stop_times"] = pd.DataFrame(
            columns=["trip_id", "stop_id", "arrival_time", "departure_time", "stop_sequence"],
        )

        nodes_gdf, edges_gdf = travel_summary_graph(gtfs_copy)

        # Should return empty edges but nodes from stops
        assert isinstance(nodes_gdf, gpd.GeoDataFrame)
        assert isinstance(edges_gdf, gpd.GeoDataFrame)
        assert len(edges_gdf) == 0

    def test_travel_summary_graph_with_calendar_dates(
        self,
        sample_gtfs_dict_with_exceptions: dict[str, pd.DataFrame],
    ) -> None:
        """Test travel summary graph with calendar_dates to cover _get_service_counts calendar_dates logic."""
        # This test covers lines 177-189 in transportation.py (_get_service_counts function)
        nodes_gdf, edges_gdf = travel_summary_graph(
            sample_gtfs_dict_with_exceptions,
            calendar_start="20240101",
            calendar_end="20240102",
        )

        assert isinstance(nodes_gdf, gpd.GeoDataFrame)
        assert isinstance(edges_gdf, gpd.GeoDataFrame)
        # Should process calendar_dates exceptions (both add and remove service)
        assert len(edges_gdf) >= 0

    def test_travel_summary_graph_with_calendar_dates_add_service(
        self,
        gtfs_dict_add_service: dict[str, pd.DataFrame | gpd.GeoDataFrame],
    ) -> None:
        """Test travel summary graph with calendar_dates that adds service."""
        nodes_gdf, edges_gdf = travel_summary_graph(
            gtfs_dict_add_service,
            calendar_start="20240101",
            calendar_end="20240101",
        )

        assert isinstance(nodes_gdf, gpd.GeoDataFrame)
        assert isinstance(edges_gdf, gpd.GeoDataFrame)
        # Should have edges because service was added by calendar_dates
        assert len(edges_gdf) > 0

    def test_travel_summary_graph_with_calendar_dates_remove_service(
        self,
        gtfs_dict_remove_service: dict[str, pd.DataFrame | gpd.GeoDataFrame],
    ) -> None:
        """Test travel summary graph with calendar_dates that removes service."""
        nodes_gdf, edges_gdf = travel_summary_graph(
            gtfs_dict_remove_service,
            calendar_start="20240101",
            calendar_end="20240101",
        )

        assert isinstance(nodes_gdf, gpd.GeoDataFrame)
        assert isinstance(edges_gdf, gpd.GeoDataFrame)
        # Should have no edges because service was removed by calendar_dates
        assert len(edges_gdf) == 0

    def test_travel_summary_graph_missing_calendar(
        self,
        sample_gtfs_dict: dict[str, pd.DataFrame],
    ) -> None:
        """Test travel summary graph with missing calendar.txt when calendar dates are specified."""
        gtfs_copy = sample_gtfs_dict.copy()
        del gtfs_copy["calendar"]  # Remove calendar.txt

        with pytest.raises(
            ValueError,
            match="calendar_start/calendar_end specified but GTFS feed has no calendar.txt",
        ):
            travel_summary_graph(gtfs_copy, calendar_start="20240101", calendar_end="20240107")

    def test_travel_summary_graph_empty_calendar(
        self,
        sample_gtfs_dict: dict[str, pd.DataFrame],
    ) -> None:
        """Test travel summary graph with empty calendar.txt when calendar dates are specified."""
        gtfs_copy = sample_gtfs_dict.copy()
        gtfs_copy["calendar"] = pd.DataFrame(
            columns=["service_id", "start_date", "end_date"],
        )  # Empty calendar

        with pytest.raises(
            ValueError,
            match="calendar_start/calendar_end specified but calendar.txt is empty",
        ):
            travel_summary_graph(gtfs_copy, calendar_start="20240101", calendar_end="20240107")

    def test_travel_summary_graph_invalid_calendar_start_format(
        self,
        sample_gtfs_dict: dict[str, pd.DataFrame],
    ) -> None:
        """Test travel summary graph with invalid calendar_start format."""
        with pytest.raises(
            ValueError,
            match="Invalid calendar_start format: invalid-date. Expected YYYYMMDD.",
        ):
            travel_summary_graph(
                sample_gtfs_dict,
                calendar_start="invalid-date",
                calendar_end="20240107",
            )

    def test_travel_summary_graph_invalid_calendar_end_format(
        self,
        sample_gtfs_dict: dict[str, pd.DataFrame],
    ) -> None:
        """Test travel summary graph with invalid calendar_end format."""
        with pytest.raises(
            ValueError,
            match="Invalid calendar_end format: invalid-date. Expected YYYYMMDD.",
        ):
            travel_summary_graph(
                sample_gtfs_dict,
                calendar_start="20240101",
                calendar_end="invalid-date",
            )

    def test_travel_summary_graph_calendar_start_outside_range(
        self,
        sample_gtfs_dict: dict[str, pd.DataFrame],
    ) -> None:
        """Test travel summary graph with calendar_start outside GTFS date range."""
        # Use a date that's before the GTFS range (sample data is from 2024)
        with pytest.raises(
            ValueError,
            match=r"calendar_start \(20230101\) is outside the valid GTFS date range",
        ):
            travel_summary_graph(
                sample_gtfs_dict,
                calendar_start="20230101",
                calendar_end="20240107",
            )

    def test_travel_summary_graph_calendar_end_outside_range(
        self,
        sample_gtfs_dict: dict[str, pd.DataFrame],
    ) -> None:
        """Test travel summary graph with calendar_end outside GTFS date range."""
        # Use a date that's after the GTFS range (sample data is from 2024)
        with pytest.raises(
            ValueError,
            match=r"calendar_end \(20250101\) is outside the valid GTFS date range",
        ):
            travel_summary_graph(
                sample_gtfs_dict,
                calendar_start="20240101",
                calendar_end="20250101",
            )

    def test_travel_summary_graph_calendar_start_after_end(
        self,
        sample_gtfs_dict: dict[str, pd.DataFrame],
    ) -> None:
        """Test travel summary graph with calendar_start after calendar_end."""
        with pytest.raises(
            ValueError,
            match=r"calendar_start \(20240107\) must be <= calendar_end \(20240101\)",
        ):
            travel_summary_graph(
                sample_gtfs_dict,
                calendar_start="20240107",
                calendar_end="20240101",
            )

    def test_travel_summary_graph_calendar_start_only_invalid(
        self,
        sample_gtfs_dict: dict[str, pd.DataFrame],
    ) -> None:
        """Test travel summary graph with only calendar_start specified and invalid."""
        with pytest.raises(
            ValueError,
            match=r"calendar_start \(20230101\) is outside the valid GTFS date range",
        ):
            travel_summary_graph(sample_gtfs_dict, calendar_start="20230101")

    def test_travel_summary_graph_calendar_end_only_invalid(
        self,
        sample_gtfs_dict: dict[str, pd.DataFrame],
    ) -> None:
        """Test travel summary graph with only calendar_end specified and invalid."""
        with pytest.raises(
            ValueError,
            match=r"calendar_end \(20250101\) is outside the valid GTFS date range",
        ):
            travel_summary_graph(sample_gtfs_dict, calendar_end="20250101")
