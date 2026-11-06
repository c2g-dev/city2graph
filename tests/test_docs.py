"""Regression checks for generated documentation and its public build hooks."""

from __future__ import annotations

import base64
import inspect
import io
import json
import re
import warnings
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pytest

import city2graph as c2g

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.pages import Page

hooks = pytest.importorskip("docs.hooks", reason="Documentation dependencies are optional")

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aD1sAAAAASUVORK5CYII="
)


@pytest.fixture
def docs_config(tmp_path: Path) -> SimpleNamespace:
    """Provide independent source and output directories."""
    source = tmp_path / "docs"
    site = tmp_path / "site"
    source.mkdir()
    site.mkdir()
    return SimpleNamespace(
        docs_dir=str(source),
        site_dir=str(site),
        site_url="https://example.org/latest/",
    )


def render(html: str, config: SimpleNamespace, *, nested: bool = False) -> str:
    """Run the public hook against a complete generated notebook page."""
    page = SimpleNamespace(
        title="Demo",
        is_homepage=False,
        file=SimpleNamespace(
            abs_src_path=str(Path(config.docs_dir) / "examples/demo.ipynb"),
            dest_uri="examples/demo/index.html" if nested else "examples/demo.html",
        ),
    )
    return cast(
        "str",
        hooks.on_post_page(
            '<article class="md-content__inner"><div class="jupyter-wrapper">'
            + html
            + "</div></article>",
            cast("Page", page),
            cast("MkDocsConfig", config),
        ),
    )


@pytest.fixture
def documentation_osm(monkeypatch: pytest.MonkeyPatch, osm_response: dict[str, Any]) -> None:
    """Serve one synthetic rail route for the live-download tutorial blocks."""
    from shapely.geometry import box  # noqa: PLC0415

    relation, stop, platform, destination, way = osm_response["elements"]
    relation["tags"].update(
        {
            "route": "subway",
            "interval": "4",
            "interval:conditional": "3 @ (Mo-Fr 07:00-09:00)",
        }
    )
    stop.update(lat=35.68, lon=139.76)
    platform.update(lat=35.6801, lon=139.76)
    destination.update(lat=35.68, lon=139.77)
    way["geometry"] = [{"lat": 35.68, "lon": 139.76}, {"lat": 35.68, "lon": 139.77}]
    raw = json.dumps(osm_response).encode()
    boundary = gpd.GeoDataFrame(geometry=[box(139.75, 35.67, 139.78, 35.69)], crs=4326)

    def geocode(place: str) -> gpd.GeoDataFrame:
        assert place == "Tokyo 23 wards, Japan"
        return boundary.copy()

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: io.BytesIO(raw))
    monkeypatch.setattr("city2graph.data.get_boundaries", geocode)
    monkeypatch.setattr(c2g, "get_boundaries", geocode)


@pytest.fixture
def tokyo_multimodal_namespace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """Execute the complete Tokyo accessibility example on disconnected street groups."""
    import networkx as nx  # noqa: PLC0415
    from shapely.geometry import Point  # noqa: PLC0415
    from shapely.geometry import Polygon  # noqa: PLC0415

    ox = pytest.importorskip("osmnx")
    guide = (Path(__file__).parents[1] / "docs/examples/creating_gtfs.md").read_text()
    namespace: dict[str, Any] = {"c2g": c2g}
    data: c2g.TransportData = {
        "routes": pd.DataFrame(
            {"route_id": ["demo"], "route_short_name": ["Demo"], "route_type": [2]}
        ),
        "stops": pd.DataFrame(
            {
                "stop_id": ["tokyo", "kanda", "akihabara"],
                "stop_name": ["Tokyo", "Kanda", "Akihabara"],
                "stop_lat": [35.6812, 35.6917, 35.6984],
                "stop_lon": [139.7671, 139.7709, 139.7731],
            }
        ),
        "patterns": pd.DataFrame(
            {"pattern_id": ["outbound"], "route_id": ["demo"], "direction_id": [0]}
        ),
        "stop_sequences": pd.DataFrame(
            {
                "pattern_id": ["outbound"] * 3,
                "stop_id": ["tokyo", "kanda", "akihabara"],
                "stop_sequence": [0, 1, 2],
                "travel_time_secs": [None, 120, 120],
            }
        ),
    }
    streets = nx.MultiDiGraph(crs="EPSG:4326", simplified=True)
    stops = data["stops"]
    for index, (_, stop) in enumerate(stops.iterrows()):
        start, end = 2 * index, 2 * index + 1
        streets.add_node(start, x=stop.stop_lon, y=stop.stop_lat)
        streets.add_node(end, x=stop.stop_lon + 0.001, y=stop.stop_lat)
        streets.add_edge(start, end, length=90)
        streets.add_edge(end, start, length=90)

    def download_streets(polygon: Polygon, **_kwargs: object) -> nx.MultiDiGraph:
        assert polygon.equals(area.geometry.union_all())
        return streets.copy()

    monkeypatch.setattr(ox, "graph_from_polygon", download_streets)
    monkeypatch.setattr(ox, "geocode", lambda _place: (35.6812, 139.7671))
    area = gpd.GeoDataFrame(geometry=[Point(139.77, 35.69).buffer(0.04)], crs=4326)
    monkeypatch.setattr(c2g, "get_boundaries", lambda _place: area.copy())
    monkeypatch.setattr("contextily.add_basemap", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.chdir(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", c2g.GTFSGenerationWarning)
        with c2g.create_gtfs(data, headway_secs=600) as con:
            namespace["con"] = con
            blocks = re.findall(
                r"<!-- gtfs-iso:\w+ -->\n[ \t]*```python\n(.*?)\n[ \t]*```", guide, re.DOTALL
            )
            assert len(blocks) == 5
            for code in blocks:
                exec(compile(dedent(code), "tokyo_isochrone_example", "exec"), namespace)  # noqa: S102
    return namespace


class TestDocumentationHooks:
    """Exercise transformations and validation through public MkDocs events."""

    def test_git_dates_require_committed_pages(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Keep new pages out of Git date rendering until they have a commit."""
        file_types = pytest.importorskip("mkdocs.structure.files")
        files = file_types.Files(
            [
                file_types.File(
                    name, str(tmp_path), str(tmp_path / "site"), use_directory_urls=False
                )
                for name in ("existing.md", "new.md", "image.png")
            ]
        )
        plugin = SimpleNamespace(config={"exclude": ["excluded.md"]})
        config = SimpleNamespace(
            docs_dir=str(tmp_path), plugins={"git-revision-date-localized": plugin}
        )
        monkeypatch.setattr(hooks.shutil, "which", lambda _name: "/usr/bin/git")
        monkeypatch.setattr(
            hooks.subprocess,
            "run",
            lambda *_args, **_kwargs: SimpleNamespace(
                returncode=0,
                stdout="existing.md\n",
            ),
        )
        assert hooks.on_files(files, cast("MkDocsConfig", config)) is files
        assert plugin.config["exclude"] == ["excluded.md", "new.md"]

    @pytest.mark.usefixtures("documentation_osm")
    def test_custom_conditions_reuse_quickstart(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Apply the visible custom conditions without a prepared input file."""
        root = Path(__file__).parents[1]
        monkeypatch.chdir(tmp_path)
        guide = (root / "docs/examples/creating_gtfs.md").read_text()
        blocks = re.findall(
            r"<!-- gtfs-example:(?:settings|osm|inspect|save) -->"
            r"\n[ \t]*```python\n(.*?)\n[ \t]*```",
            guide,
            flags=re.DOTALL,
        )
        assert len(blocks) == 4
        namespace: dict[str, Any] = {"c2g": c2g}
        for block in blocks:
            exec(compile(dedent(block), "tokyo_custom", "exec"), namespace)  # noqa: S102
        namespace["configured_con"].close()
        with c2g.load_gtfs(tmp_path / "synthetic.zip") as restored:
            report = c2g.get_gtfs_summary(restored)
            assert report["summary"].iloc[0]["pattern_count"] == 1
            assert report["summary"].iloc[0]["service_days"] == 7
            assert report["services"]["headway_secs"].eq(600).all()
            assert report["services"]["start_time"].eq("07:00:00").all()
            assert report["services"]["end_time"].eq("09:00:00").all()

    @pytest.mark.usefixtures("documentation_osm")
    def test_introductory_tokyo_code(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Run the OSM quick start and its complete map against a synthetic source."""
        root = Path(__file__).parents[1]
        monkeypatch.setattr("contextily.add_basemap", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(plt, "show", lambda: None)
        monkeypatch.chdir(tmp_path)
        guide = (root / "docs/examples/creating_gtfs.md").read_text()
        blocks = re.findall(
            r"<!-- gtfs-quick:osm(?:-graph|-plot)? -->\n[ \t]*```python\n(.*?)\n[ \t]*```",
            guide,
            flags=re.DOTALL,
        )
        assert len(blocks) == 3
        namespace: dict[str, Any] = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", c2g.GTFSGenerationWarning)
            for block in blocks:
                exec(compile(dedent(block), "tokyo_intro", "exec"), namespace)  # noqa: S102
        with namespace["con"] as con:
            report = c2g.get_gtfs_summary(con)
            assert report["summary"].iloc[0]["pattern_count"] == 1
            assert report["summary"].iloc[0]["service_days"] == 7
            services = report["services"].query("pattern_id == 'osm:relation:42'")
            assert {180, 240}.issubset(services["headway_secs"])
            assert report["summary"].iloc[0]["excluded_relation_count"] == 0
        assert len(namespace["nodes"]) == 2
        assert len(namespace["edges"]) == 1
        assert namespace["edges"]["frequency"].gt(0).all()
        assert namespace["ax"].get_legend().get_title().get_text() == "Modelled traversals/day"
        labels = [text.get_text() for text in namespace["ax"].get_legend().get_texts()]
        assert labels == [
            "[0, 250)",
            "[250, 500)",
            "[500, 1,000)",
            "[1,000, 2,000)",
            "[2,000, 4,000)",
            "Connected stations",
            "1\u20132",
            "3",
            "4",
            "5 or more",
        ]
        plt.close("all")

    def test_tokyo_multimodal_example(self, tokyo_multimodal_namespace: dict[str, Any]) -> None:
        """Reach separate walking components by rail without filling rail corridors."""
        from shapely.geometry import Point  # noqa: PLC0415

        namespace = tokyo_multimodal_namespace
        assert len(namespace["access_edges"]) == 2
        assert namespace["multimodal"].is_directed()
        access = [
            attrs
            for _, _, attrs in namespace["multimodal"].edges(data=True)
            if attrs.get("edge_type") == ("street", "is_nearby", "station")
        ]
        assert len(access) == 6
        assert sum(edge["travel_time_sec"] == 180 for edge in access) == 3
        assert all(edge["travel_time_sec"] >= 180 for edge in access)
        polygons = namespace["isochrones"]
        assert polygons["threshold"].tolist() == [600, 1200, 1800]
        assert polygons.is_valid.all()
        assert not polygons.is_empty.any()
        destination = gpd.GeoSeries([Point(139.7731, 35.6984)], crs=4326).to_crs(6677).iloc[0]
        assert polygons.geometry.iloc[0].covers(destination)
        walk_graph = c2g.gdf_to_nx(namespace["walk_nodes"], namespace["walk_edges"], directed=True)
        walking = c2g.create_isochrone(
            graph=walk_graph,
            center_point=namespace["center"],
            threshold=1800,
            edge_attr="travel_time_sec",
            method="buffer",
            buffer_distance=40,
        )
        assert not walking.geometry.iloc[0].covers(destination)
        midpoint = (
            gpd.GeoSeries([Point((139.7671 + 139.7709) / 2, (35.6812 + 35.6917) / 2)], crs=4326)
            .to_crs(6677)
            .iloc[0]
        )
        assert not polygons.geometry.iloc[-1].covers(midpoint)
        boundary = namespace["area"].geometry.union_all()
        assert polygons.geometry.difference(boundary).area.sum() < 1e-4
        ax = namespace["ax"]
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        assert xlim[1] - xlim[0] == pytest.approx(ylim[1] - ylim[0])
        xmin, ymin, xmax, ymax = namespace["area"].to_crs(3857).total_bounds
        assert xlim[0] < xmin < xmax < xlim[1]
        assert ylim[0] < ymin < ymax < ylim[1]
        assert ax.get_title() == ""
        plt.close("all")

    def test_gtfs_registry_tables_and_executable_examples(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Keep published defaults and copyable offline examples aligned with runtime."""
        root = Path(__file__).parents[1]
        registry = json.loads((root / "city2graph/gtfs/defaults.json").read_text())
        guide = (root / "docs/api/gtfs_generation.md").read_text()
        output = hooks.on_page_markdown(guide)
        assert "<!-- gtfs-defaults:" not in output
        for entry in registry["settings"].values():
            value = (
                json.dumps(entry["value"])
                if isinstance(entry["value"], bool)
                else str(entry["value"])
            )
            assert f"`{value}` ({entry['unit']})" in output
        assert registry["model"] in output
        article = hooks.on_page_markdown((root / "docs/examples/creating_gtfs.md").read_text())
        assert "<!-- gtfs-defaults:" not in article
        assert f"shared {registry['settings']['group_headway_secs']['value']} s interval" in article
        for function in (c2g.create_gtfs, c2g.load_osm_transport):
            for parameter in inspect.signature(function).parameters.values():
                if parameter.kind != inspect.Parameter.VAR_KEYWORD:
                    assert f"`{parameter.name}`" in article
        namespace: dict[str, Any] = {"__name__": "__docs_example__"}
        examples = re.findall(
            r"<!-- gtfs-example:offline -->\n```python\n(.*?)```"
            r"(?:\n\n<!-- gtfs-output:start -->\n(.*?)\n<!-- gtfs-output:end -->)?",
            guide,
            flags=re.DOTALL,
        )
        assert examples, "The GTFS guide must contain executable offline examples"
        assert len(examples) == guide.count("<!-- gtfs-example:offline -->")
        displayed: list[pd.DataFrame] = []
        monkeypatch.setattr("IPython.display.display", displayed.append)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", c2g.GTFSGenerationWarning)
                for index, (source, expected) in enumerate(examples):
                    block = source.replace('"synthetic.zip"', repr(str(tmp_path / "synthetic.zip")))
                    displayed.clear()
                    exec(compile(block, f"creating_gtfs_example_{index}", "exec"), namespace)  # noqa: S102
                    actual = "\n\n".join(
                        frame.to_html(border=0).replace(' class="dataframe"', "")
                        for frame in displayed
                    )
                    assert actual.rstrip() == expected.rstrip(), f"Example {index}"
                    if 'con = c2g.create_gtfs(data, service_date="20261104")' in block:
                        report = c2g.get_gtfs_summary(namespace["con"])
                        adopted = report["assumptions"]
                        assert set(adopted.query("field == 'speed_kph'")["value"]) == {
                            str(registry["speeds_kph"]["3"])
                        }
                        assert set(report["services"]["headway_secs"]) == {
                            registry["settings"]["group_headway_secs"]["value"]
                        }
                        assert report["services"].date.min() == "20261104"
                        assert report["summary"].loc[0, "pattern_count"] == 1
                        assert report["summary"].loc[0, "service_days"] == 1
                        assert report["edges"].frequency.eq(64).all()
                        assert not report["edges"].empty
                    if "configured_report = c2g.get_gtfs_summary(" in block:
                        assert (
                            namespace["configured_summary"].iloc[0]["mean_daily_departures"] == 12
                        )
                    if "weekday_report = c2g.get_gtfs_summary(" in block:
                        services = namespace["weekday_report"]["services"]
                        active = services.query("departures > 0")["date"].tolist()
                        assert active == ["20261102", "20261103", "20261105", "20261106"]
            assert (tmp_path / "synthetic.zip").is_file()
        finally:
            for value in namespace.values():
                if isinstance(value, duckdb.DuckDBPyConnection):
                    value.close()

    @pytest.mark.parametrize("nested", [False, True])
    def test_notebook_images_are_cacheable_and_lossless(
        self, docs_config: SimpleNamespace, nested: bool
    ) -> None:
        """Extract identical PNGs once without changing image bytes or captions."""
        data_url = "data:image/png;base64," + base64.b64encode(PNG).decode()
        html = f'<p>Plot &amp; caption</p><img src="{data_url}" alt="A &amp; B">'
        output = render(html + html, docs_config, nested=nested)
        images = list(Path(docs_config.site_dir).rglob("*.png"))
        assert len(images) == 1
        assert images[0].read_bytes() == PNG
        assert "data:image" not in output
        assert output.count('loading="lazy"') == 1
        assert output.count('loading="eager"') == 1
        assert output.count('decoding="async"') == 2
        assert output.count('width="1"') == 2
        assert output.count('height="1"') == 2
        assert 'alt="A &amp; B"' in output
        prefix = "../../assets/" if nested else "../assets/"
        assert f'src="{prefix}generated/' in output
        assert render(html + html, docs_config, nested=nested) == output

    def test_only_obsolete_notebook_scripts_are_removed(self, docs_config: SimpleNamespace) -> None:
        """Keep executable notebook code while removing incompatible MathJax setup."""
        script = "<script>const example = '<img src=\"demo.png\">';</script>"
        output = render(
            script + '<script src=""></script>'
            '<script type="text/x-mathjax-config">MathJax.Hub.Config({});</script>'
            '<script src="plot.js"></script><video src="plot.mp4"></video>',
            docs_config,
        )
        assert script in output
        assert '<script src="plot.js"></script>' in output
        assert 'src=""' not in output
        assert "text/x-mathjax-config" not in output
        assert 'preload="none"' in output

    def test_common_css_is_shared_and_invalid_duplicate_is_removed(
        self, docs_config: SimpleNamespace
    ) -> None:
        """Share valid stylesheets and discard the invalid repeated Python list."""
        css = ".jupyter-wrapper {color: red;}" * 100
        html = f"<style>{css}</style><style>{[css]!r}</style>"
        first = render(html, docs_config)
        second = render(html, docs_config)
        stylesheets = list(Path(docs_config.site_dir).rglob("*.css"))
        assert len(stylesheets) == 1
        assert stylesheets[0].read_text() == css
        assert "<style>" not in first
        assert first == second

    @pytest.mark.parametrize("has_view", [False, True])
    def test_widget_state_is_kept_only_when_used(
        self, docs_config: SimpleNamespace, has_view: bool
    ) -> None:
        """Preserve widget data only when a rendered widget references it."""
        html = '<script type="application/vnd.jupyter.widget-state+json">{"state": {}}</script>'
        if has_view:
            html += (
                '<script type="application/vnd.jupyter.widget-view+json">{"model_id": "a"}</script>'
            )
        output = render(html, docs_config)
        assert ("widget-state" in output) is has_view

    @pytest.mark.parametrize(
        "source",
        [
            "data:image/png;base64,not-valid!",
            "data:image/svg+xml;base64,PHN2Zy8+",
            "https://example.org/plot.png",
        ],
    )
    def test_other_images_are_not_rewritten(
        self, docs_config: SimpleNamespace, source: str
    ) -> None:
        """Leave unsupported and external image URLs intact."""
        output = render(f'<img src="{source}">', docs_config)
        assert f'src="{source}"' in output
        assert not list(Path(docs_config.site_dir).rglob("*.png"))

    def test_local_image_dimensions_prevent_layout_shift(
        self, docs_config: SimpleNamespace
    ) -> None:
        """Read dimensions from referenced local raster files."""
        Path(docs_config.docs_dir, "plot.png").write_bytes(PNG)
        output = render('<img src="../plot.png">', docs_config)
        assert 'width="1"' in output
        assert 'height="1"' in output
        sized = render('<img src="../plot.png" width="40">', docs_config)
        assert 'height="40"' in sized

    def test_valid_site_links_include_subpaths_and_encoded_anchors(
        self, docs_config: SimpleNamespace
    ) -> None:
        """Accept existing files under the deployment prefix and encoded IDs."""
        root = Path(docs_config.site_dir)
        (root / "guide").mkdir()
        (root / "guide/index.html").write_text('<h1 id="\u03b1">Guide</h1>')
        (root / "plot.png").write_bytes(PNG)
        (root / "index.html").write_text(
            '<a href="guide/#%CE%B1">Guide</a>'
            '<a href="/latest/guide/index.html#\u03b1">Guide</a>'
            '<img src="plot.png"><a href="https://example.org/elsewhere">External</a>'
            '<a href="mailto:author@example.org">Email</a><a href="/other-version/">Other version</a>'
        )
        (root / "404.html").write_text('<a href="/latest/index.html">Home</a>')
        (root / ".DS_Store").write_bytes(b"Finder metadata")
        hooks.on_post_build(docs_config)
        assert not (root / ".DS_Store").exists()

    def test_plot_context_replaces_placeholder_but_preserves_decorative_alt(
        self, docs_config: SimpleNamespace
    ) -> None:
        """Give exported plots context while keeping explicit decorative images silent."""
        Path(docs_config.docs_dir, "plot.png").write_bytes(PNG)
        output = render(
            "<h2>Accessibility graph</h2>"
            '<img src="../plot.png" alt="No description has been provided for this image">'
            '<img src="../plot.png" alt="">',
            docs_config,
        )
        assert 'alt="Plot: Accessibility graph"' in output
        assert 'alt=""' in output

    @pytest.mark.parametrize(
        "markup",
        [
            '<a href="missing.html">Page</a>',
            '<img src="missing.png">',
            '<script src="missing.js"></script>',
            '<link rel="stylesheet" href="missing.css">',
            '<iframe src="missing.pdf"></iframe>',
            '<video><source src="missing.mp4"></video>',
            '<video poster="missing.jpg"></video>',
            '<a href="index.html#missing">Anchor</a>',
            '<source srcset="missing.webp 2x">',
        ],
    )
    def test_broken_references_fail_the_build(
        self, docs_config: SimpleNamespace, markup: str
    ) -> None:
        """Reject missing HTML targets and static resources."""
        Path(docs_config.site_dir, "index.html").write_text(markup)
        with pytest.raises(RuntimeError, match=r"index\.html: missing"):
            hooks.on_post_build(docs_config)

    def test_missing_css_assets_fail_the_build(self, docs_config: SimpleNamespace) -> None:
        """Validate resource URLs within generated stylesheets."""
        Path(docs_config.site_dir, "theme.css").write_text("body {background: url('missing.png')}")
        with pytest.raises(RuntimeError, match=r"theme\.css: missing file"):
            hooks.on_post_build(docs_config)
