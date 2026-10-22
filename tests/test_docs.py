"""Regression checks for generated documentation and its public build hooks."""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast

import pytest

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


class TestDocumentationHooks:
    """Exercise transformations and validation through public MkDocs events."""

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
