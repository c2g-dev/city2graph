"""
OSM and SUMO network file import utilities.

Functions in this module load road network files from OpenStreetMap (.osm, .osm.gz)
and SUMO (.net.xml) formats into GeoDataFrames and optionally NetworkX graphs,
following the same (nodes_gdf, edges_gdf) convention used throughout city2graph.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString
from shapely.geometry import Point

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = ["load_osm_network", "load_sumo_network"]


def _network_gdfs_to_nx(
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    *,
    directed: bool,
) -> nx.DiGraph | nx.Graph:
    graph: nx.DiGraph | nx.Graph = nx.DiGraph() if directed else nx.Graph()
    graph.graph["crs"] = str(nodes_gdf.crs) if nodes_gdf.crs else None
    for node_id, row in nodes_gdf.iterrows():
        graph.add_node(node_id, **row.to_dict())
    for (src, dst), row in edges_gdf.iterrows():
        graph.add_edge(src, dst, **row.to_dict())
    return graph


# ---------------------------------------------------------------------------
# OSM
# ---------------------------------------------------------------------------


def _osmnx_to_gdfs(
    G: nx.MultiDiGraph,
    *,
    directed: bool,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    import osmnx as ox

    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
    nodes_gdf = nodes_gdf.to_crs("EPSG:4326")
    edges_gdf = edges_gdf.to_crs("EPSG:4326")

    # Drop the osmnx 'key' level from the (u, v, key) MultiIndex.
    edges_gdf = edges_gdf.reset_index()

    if directed:
        edges_gdf = edges_gdf.drop_duplicates(subset=["u", "v"]).set_index(["u", "v"])
    else:
        # Canonicalize direction so u <= v, then deduplicate.
        swap = edges_gdf["u"] > edges_gdf["v"]
        edges_gdf.loc[swap, ["u", "v"]] = edges_gdf.loc[swap, ["v", "u"]].to_numpy()
        edges_gdf = edges_gdf.drop_duplicates(subset=["u", "v"]).set_index(["u", "v"])

    return nodes_gdf, edges_gdf


def load_osm_network(
    path: str | Path,
    *,
    retain_all: bool = True,
    simplify: bool = True,
    as_nx: bool = False,
    directed: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.DiGraph | nx.Graph:
    """
    Load an OpenStreetMap network file into GeoDataFrames or a NetworkX graph.

    Supports ``.osm`` and ``.osm.gz`` formats via osmnx. The result follows
    the same ``(nodes_gdf, edges_gdf)`` convention used throughout city2graph.

    Parameters
    ----------
    path : str | Path
        Path to an OSM file (``.osm`` or ``.osm.gz``).
    retain_all : bool, default=True
        If ``True``, retain all graph components including disconnected islands.
    simplify : bool, default=True
        If ``True``, simplify graph topology by removing interstitial nodes.
    as_nx : bool, default=False
        If ``True``, return a NetworkX graph instead of GeoDataFrames.
    directed : bool, default=False
        If ``True``, return a directed graph preserving OSM edge direction.
        If ``False``, each undirected road is represented once with
        ``u <= v`` canonical ordering.

    Returns
    -------
    tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.DiGraph | nx.Graph
        ``(nodes_gdf, edges_gdf)`` when ``as_nx`` is ``False``; otherwise a
        ``nx.DiGraph`` (``directed=True``) or ``nx.Graph`` (``directed=False``).
        Nodes are indexed by OSM node id. Edges are indexed by ``(u, v)``.
        Both GeoDataFrames use CRS ``EPSG:4326``.

    Raises
    ------
    ImportError
        If osmnx is not installed.
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file cannot be parsed as a valid OSM network.
    """
    try:
        import osmnx as ox
    except ImportError as exc:
        msg = "osmnx is required for load_osm_network. Install it with: pip install osmnx"
        raise ImportError(msg) from exc

    file_path = Path(path)
    if not file_path.exists():
        msg = f"OSM file not found: {file_path}"
        raise FileNotFoundError(msg)

    try:
        G: nx.MultiDiGraph = ox.graph_from_xml(
            file_path,
            retain_all=retain_all,
            simplify=simplify,
            bidirectional=directed,
        )
    except Exception as exc:
        msg = f"Failed to parse OSM file {file_path}: {exc}"
        raise ValueError(msg) from exc

    nodes_gdf, edges_gdf = _osmnx_to_gdfs(G, directed=directed)

    if as_nx:
        return _network_gdfs_to_nx(nodes_gdf, edges_gdf, directed=directed)
    return nodes_gdf, edges_gdf


# ---------------------------------------------------------------------------
# SUMO
# ---------------------------------------------------------------------------


def _parse_sumo_location(
    location_elem: ET.Element | None,
) -> tuple[float, float, str | None]:
    """
    Extract coordinate offset and projection string from a SUMO ``<location>`` element.

    Parameters
    ----------
    location_elem : ET.Element | None
        The ``<location>`` XML element, or ``None`` when absent.

    Returns
    -------
    tuple[float, float, str | None]
        ``(offset_x, offset_y, proj_string)`` where offsets are the values from
        ``netOffset`` and ``proj_string`` is ``None`` when unavailable or ``"!"``.
    """
    if location_elem is None:
        return 0.0, 0.0, None

    net_offset = location_elem.get("netOffset", "0.00,0.00")
    proj_param = location_elem.get("projParameter", "")

    try:
        ox_str, oy_str = net_offset.split(",")
        offset_x, offset_y = float(ox_str), float(oy_str)
    except (ValueError, AttributeError):
        offset_x, offset_y = 0.0, 0.0

    proj_string = proj_param.strip() if proj_param.strip() not in {"", "!"} else None
    return offset_x, offset_y, proj_string


def _sumo_xy_to_lonlat(
    x: float,
    y: float,
    *,
    offset_x: float,
    offset_y: float,
    proj_string: str | None,
) -> tuple[float, float]:
    """
    Convert a SUMO local ``(x, y)`` coordinate to WGS84 ``(lon, lat)``.

    SUMO net coordinates satisfy ``sumo = projected - netOffset``, so the
    inverse is ``projected = sumo + netOffset``, followed by the inverse
    cartographic projection.

    Parameters
    ----------
    x : float
        SUMO net x-coordinate (metres in local CRS).
    y : float
        SUMO net y-coordinate (metres in local CRS).
    offset_x : float
        ``netOffset`` x component.
    offset_y : float
        ``netOffset`` y component.
    proj_string : str | None
        PROJ projection string embedded in the SUMO file, or ``None`` when
        no projection is defined (coordinates treated as lon/lat directly).

    Returns
    -------
    tuple[float, float]
        ``(lon, lat)`` in WGS84 degrees.
    """
    proj_x = x + offset_x
    proj_y = y + offset_y

    if proj_string is None:
        return proj_x, proj_y

    try:
        from pyproj import Proj
        from pyproj import Transformer

        src_proj = Proj(proj_string)
        transformer = Transformer.from_proj(src_proj, Proj("EPSG:4326"), always_xy=True)
        lon, lat = transformer.transform(proj_x, proj_y)
        return float(lon), float(lat)
    except Exception:
        logger.warning(
            "SUMO coordinate reprojection failed for proj=%r; returning offset-corrected values.",
            proj_string,
        )
        return proj_x, proj_y


def _parse_sumo_shape(
    shape_str: str,
    *,
    offset_x: float,
    offset_y: float,
    proj_string: str | None,
) -> LineString | None:
    """
    Parse a SUMO ``shape`` attribute string into a reprojected ``LineString``.

    Parameters
    ----------
    shape_str : str
        Space-separated ``"x,y"`` pairs from a SUMO shape attribute.
    offset_x : float
        ``netOffset`` x component.
    offset_y : float
        ``netOffset`` y component.
    proj_string : str | None
        PROJ string for reprojection, or ``None``.

    Returns
    -------
    LineString | None
        Reprojected geometry, or ``None`` when fewer than two valid points exist.
    """
    if not shape_str:
        return None
    try:
        coords = []
        for pair in shape_str.strip().split():
            if "," not in pair:
                continue
            px, py = pair.split(",", 1)
            lon, lat = _sumo_xy_to_lonlat(
                float(px),
                float(py),
                offset_x=offset_x,
                offset_y=offset_y,
                proj_string=proj_string,
            )
            coords.append((lon, lat))
        return LineString(coords) if len(coords) >= 2 else None  # noqa: TRY300
    except (ValueError, IndexError):
        return None


def _safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _build_sumo_nodes(
    root: ET.Element,
    *,
    offset_x: float,
    offset_y: float,
    proj_string: str | None,
) -> gpd.GeoDataFrame:
    """
    Extract SUMO junctions into a node GeoDataFrame.

    Internal junctions (``type="internal"``) are skipped.

    Parameters
    ----------
    root : ET.Element
        Root element of the parsed SUMO net XML.
    offset_x : float
        ``netOffset`` x component.
    offset_y : float
        ``netOffset`` y component.
    proj_string : str | None
        PROJ string for reprojection, or ``None``.

    Returns
    -------
    gpd.GeoDataFrame
        Node records indexed by junction id, CRS ``EPSG:4326``.
    """
    records = []
    for junc in root.iter("junction"):
        if junc.get("type") == "internal":
            continue
        junc_id = junc.get("id", "")
        try:
            raw_x = float(junc.get("x", "0"))
            raw_y = float(junc.get("y", "0"))
        except ValueError:
            continue
        lon, lat = _sumo_xy_to_lonlat(
            raw_x,
            raw_y,
            offset_x=offset_x,
            offset_y=offset_y,
            proj_string=proj_string,
        )
        records.append(
            {
                "node_id": junc_id,
                "junction_type": junc.get("type", ""),
                "geometry": Point(lon, lat),
            }
        )

    if not records:
        return gpd.GeoDataFrame(
            columns=["node_id", "junction_type", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        ).set_index("node_id")

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    return gdf.set_index("node_id")


def _build_sumo_edges(
    root: ET.Element,
    *,
    offset_x: float,
    offset_y: float,
    proj_string: str | None,
    nodes_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Extract SUMO edges into an edge GeoDataFrame.

    Internal junction connector edges (id starting with ``":"``) are skipped.
    Lane-level speed and length are averaged across lanes per edge.
    When no explicit shape is present, a straight line between the endpoint
    junctions is used.

    Parameters
    ----------
    root : ET.Element
        Root element of the parsed SUMO net XML.
    offset_x : float
        ``netOffset`` x component.
    offset_y : float
        ``netOffset`` y component.
    proj_string : str | None
        PROJ string for reprojection, or ``None``.
    nodes_gdf : gpd.GeoDataFrame
        Node GeoDataFrame used for fallback straight-line geometries.

    Returns
    -------
    gpd.GeoDataFrame
        Edge records indexed by ``(from_id, to_id)``, CRS ``EPSG:4326``.
    """
    records = []
    for edge in root.iter("edge"):
        edge_id = edge.get("id", "")
        if edge_id.startswith(":"):
            continue
        from_id = edge.get("from", "")
        to_id = edge.get("to", "")
        if not from_id or not to_id:
            continue

        shape_str = edge.get("shape", "")
        geom = _parse_sumo_shape(
            shape_str,
            offset_x=offset_x,
            offset_y=offset_y,
            proj_string=proj_string,
        )
        if geom is None and from_id in nodes_gdf.index and to_id in nodes_gdf.index:
            geom = LineString(
                [
                    nodes_gdf.loc[from_id, "geometry"],
                    nodes_gdf.loc[to_id, "geometry"],
                ]
            )

        lanes = edge.findall("lane")
        num_lanes = len(lanes)
        speeds = [float(ln.get("speed")) for ln in lanes if ln.get("speed")]
        lengths = [float(ln.get("length")) for ln in lanes if ln.get("length")]

        records.append(
            {
                "from_id": from_id,
                "to_id": to_id,
                "edge_id": edge_id,
                "num_lanes": num_lanes,
                "speed_ms": sum(speeds) / len(speeds) if speeds else None,
                "length_m": sum(lengths) / len(lengths) if lengths else None,
                "priority": _safe_int(edge.get("priority")),
                "edge_type": edge.get("type", ""),
                "geometry": geom,
            }
        )

    if not records:
        return gpd.GeoDataFrame(
            columns=[
                "from_id",
                "to_id",
                "edge_id",
                "num_lanes",
                "speed_ms",
                "length_m",
                "priority",
                "edge_type",
                "geometry",
            ],
            geometry="geometry",
            crs="EPSG:4326",
        ).set_index(["from_id", "to_id"])

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    return gdf.set_index(["from_id", "to_id"])


def load_sumo_network(
    path: str | Path,
    *,
    as_nx: bool = False,
    directed: bool = True,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.DiGraph | nx.Graph:
    """
    Load a SUMO network file (``.net.xml``) into GeoDataFrames or a NetworkX graph.

    SUMO networks use a local Cartesian coordinate system. When the
    ``<location>`` element contains a valid ``projParameter`` attribute,
    coordinates are reprojected to WGS84 (EPSG:4326) via pyproj (available
    as a transitive dependency through geopandas). When no projection is
    defined (``projParameter="!"``), offset-corrected coordinates are returned
    directly and are expected to already be in degrees.

    Parameters
    ----------
    path : str | Path
        Path to a SUMO network file (``.net.xml``).
    as_nx : bool, default=False
        If ``True``, return a NetworkX graph instead of GeoDataFrames.
    directed : bool, default=True
        If ``True``, return a directed graph preserving SUMO edge direction.
        SUMO networks are inherently directed; setting this to ``False``
        canonicalises each pair so that ``from_id <= to_id`` and keeps one
        representative edge per undirected pair.

    Returns
    -------
    tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.DiGraph | nx.Graph
        ``(nodes_gdf, edges_gdf)`` when ``as_nx`` is ``False``; otherwise a
        ``nx.DiGraph`` (``directed=True``) or ``nx.Graph`` (``directed=False``).
        Nodes are indexed by junction id. Edges are indexed by
        ``(from_id, to_id)``. Both GeoDataFrames use CRS ``EPSG:4326``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file cannot be parsed as valid SUMO XML.
    """
    file_path = Path(path)
    if not file_path.exists():
        msg = f"SUMO network file not found: {file_path}"
        raise FileNotFoundError(msg)

    try:
        tree = ET.parse(file_path)  # noqa: S314
        root = tree.getroot()
    except ET.ParseError as exc:
        msg = f"Failed to parse SUMO network file {file_path}: {exc}"
        raise ValueError(msg) from exc

    offset_x, offset_y, proj_string = _parse_sumo_location(root.find("location"))

    nodes_gdf = _build_sumo_nodes(
        root,
        offset_x=offset_x,
        offset_y=offset_y,
        proj_string=proj_string,
    )
    edges_gdf = _build_sumo_edges(
        root,
        offset_x=offset_x,
        offset_y=offset_y,
        proj_string=proj_string,
        nodes_gdf=nodes_gdf,
    )

    if not directed and not edges_gdf.empty:
        edges_gdf = edges_gdf.reset_index()
        swap = edges_gdf["from_id"] > edges_gdf["to_id"]
        edges_gdf.loc[swap, ["from_id", "to_id"]] = (
            edges_gdf.loc[swap, ["to_id", "from_id"]].to_numpy()
        )
        edges_gdf = edges_gdf.drop_duplicates(subset=["from_id", "to_id"]).set_index(
            ["from_id", "to_id"]
        )

    if as_nx:
        return _network_gdfs_to_nx(nodes_gdf, edges_gdf, directed=directed)
    return nodes_gdf, edges_gdf
