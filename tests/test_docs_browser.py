"""Chromium smoke tests, enabled explicitly against a completed MkDocs build."""

from __future__ import annotations

import io
import os
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import BinaryIO

    from playwright.sync_api import Browser
    from playwright.sync_api import Page

SITE = os.environ.get("CITY2GRAPH_DOCS_SITE")
pytestmark = pytest.mark.skipif(not SITE, reason="Set CITY2GRAPH_DOCS_SITE to a built site")


@pytest.fixture(scope="module")
def docs_server() -> Iterator[str]:
    """Serve the build under the production /latest/ mount point."""
    assert SITE
    root = Path(SITE).resolve()
    assert (root / "index.html").is_file()

    class Handler(SimpleHTTPRequestHandler):
        def translate_path(self, path: str) -> str:
            return super().translate_path(path.removeprefix("/latest"))

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            pass

        def send_head(self) -> BinaryIO | None:
            # Material uses the sitemap to recognise instant-navigation targets.
            # Match mkdocs serve by giving local URLs the test server's port.
            if self.path.partition("?")[0] == "/latest/sitemap.xml":
                sitemap = (
                    (root / "sitemap.xml")
                    .read_text()
                    .replace(
                        "https://city2graph.net",
                        "http://" + self.headers["Host"],
                    )
                    .encode()
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/xml")
                self.send_header("Content-Length", str(len(sitemap)))
                self.end_headers()
                return io.BytesIO(sitemap)
            return super().send_head()

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(root)))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield f"http://127.0.0.1:{server.server_port}/latest/"
    server.shutdown()
    server.server_close()
    worker.join()


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    """Launch Chromium only for the explicitly requested documentation checks."""
    api = pytest.importorskip("playwright.sync_api")
    with api.sync_playwright() as playwright:
        instance = playwright.chromium.launch()
        yield instance
        instance.close()


@pytest.fixture
def page(browser: Browser) -> Iterator[Page]:
    """Avoid sending test page views to the production analytics property."""
    context = browser.new_context(permissions=["clipboard-read", "clipboard-write"])
    context.route("https://www.googletagmanager.com/**", lambda route: route.fulfill(body=""))
    page = context.new_page()
    page.set_default_timeout(15000)
    yield page
    context.close()


def test_navigation_search_copy_and_analytics(page: Page, docs_server: str) -> None:
    """Keep normal pages and instant navigation functional with one page view each."""
    errors: list[str] = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(docs_server, wait_until="domcontentloaded")
    page.wait_for_function("typeof document$ !== 'undefined'")
    page.wait_for_function(
        "performance.getEntriesByType('resource').some(entry => entry.name.endsWith('/sitemap.xml'))"
    )
    page.evaluate("window.docsNavigationSentinel = true")
    assert page.locator('a[href$="paper.html"] img').count() == 1
    assert page.locator('script[src*="mathjax@"]').count() == 0
    page.locator('.md-tabs a[href$="installation.html"]').click()
    page.wait_for_url("**/installation.html")
    page.get_by_role("heading", name="Install City2Graph", exact=False).first.wait_for()
    assert page.evaluate("window.docsNavigationSentinel") is True
    # Material closes search 125 ms after navigation, after the new DOM is visible.
    # Let that transition finish before focusing the next interaction.
    page.wait_for_timeout(150)
    page.locator(".md-search__input").click()
    page.locator(".md-search__input").press_sequentially("morphological_graph")
    page.locator(".md-search-result__link").first.wait_for()
    assert page.locator(".md-search-result__link").count() > 0
    page.keyboard.press("Escape")
    page.get_by_role("button", name="Copy to clipboard").first.click()
    page.wait_for_function(
        "navigator.clipboard.readText().then(text => text.includes('pip install'))"
    )
    page.go_back()
    page.wait_for_url(docs_server)
    page.get_by_role("heading", name="City2Graph: Geospatial Graphs", exact=False).first.wait_for()
    records = page.evaluate("Array.from(window.dataLayer || [], row => Array.from(row))")
    assert [row[0] for row in records[:3]] == ["consent", "js", "config"]
    assert records[0][2]["analytics_storage"] == "denied"
    assert len([row for row in records if row[0] == "config"]) == 1
    views = [row for row in records if row[:2] == ["event", "page_view"]]
    assert [row[2]["page_path"] for row in views] == [
        "/latest/",
        "/latest/installation.html",
        "/latest/",
    ]
    assert not errors


def test_version_selector_keyboard_and_repeated_events(page: Page, docs_server: str) -> None:
    """Handle delayed Read the Docs data without duplicating version controls."""
    page.goto(docs_server, wait_until="domcontentloaded")
    page.wait_for_function("typeof document$ !== 'undefined'")
    page.wait_for_function(
        "performance.getEntriesByType('resource').some(entry => entry.name.endsWith('/sitemap.xml'))"
    )
    page.evaluate("""() => {
      const versions = {current: {slug: 'latest'}, active: [
        {slug: 'latest', urls: {documentation: '/latest/'}},
        {slug: 'stable', urls: {documentation: '/stable/'}}
      ]};
      for (let i = 0; i < 2; i++) document.dispatchEvent(new CustomEvent(
        'readthedocs-addons-data-ready', {detail: {data: () => ({versions})}}
      ));
    }""")
    button = page.get_by_role("button", name="Select version")
    assert button.count() == 1
    button.focus()
    page.keyboard.press("Enter")
    assert button.get_attribute("aria-expanded") == "true"
    assert page.locator(".md-version__link").first.evaluate(
        "node => node === document.activeElement"
    )
    page.keyboard.press("Escape")
    assert button.get_attribute("aria-expanded") == "false"
    assert button.evaluate("node => node === document.activeElement")
    page.locator('.md-tabs a[href$="installation.html"]').click()
    page.get_by_role("heading", name="Install City2Graph", exact=False).first.wait_for()
    assert button.count() == 1


def test_notebook_copy_after_instant_navigation(page: Page, docs_server: str) -> None:
    """Initialise notebook controls when a tutorial is opened without a reload."""
    page.goto(docs_server, wait_until="domcontentloaded")
    page.wait_for_function(
        "performance.getEntriesByType('resource').some(entry => entry.name.endsWith('/sitemap.xml'))"
    )
    page.evaluate("window.docsNavigationSentinel = true")
    page.locator('.examples-gallery a[href$="add_metapaths.html"]').click()
    page.locator(".jupyter-wrapper").wait_for()
    assert page.evaluate("window.docsNavigationSentinel") is True
    page.locator("clipboard-copy").first.click()
    page.wait_for_function("navigator.clipboard.readText().then(text => text.includes('import'))")
    page.go_back()
    page.get_by_role("heading", name="City2Graph: Geospatial Graphs", exact=False).first.wait_for()
    page.locator('.examples-gallery a[href$="add_metapaths.html"]').click()
    page.locator("clipboard-copy").first.click()
    page.wait_for_function("navigator.clipboard.readText().then(text => text.includes('import'))")


def test_not_found_page_resolves_versioned_assets(page: Page, docs_server: str) -> None:
    """Render the 404 document with the assets from its versioned mount."""
    page.goto(docs_server + "404.html", wait_until="domcontentloaded")
    page.wait_for_function("document.querySelector('.md-header img').naturalWidth > 0")
    assert page.locator(".md-content").is_visible()
    assert page.evaluate("getComputedStyle(document.body).fontFamily").startswith("Inter")


def test_site_payload_budget_and_exclusions() -> None:
    """Keep the largest tutorial small and avoid publishing source-only files."""
    assert SITE
    root = Path(SITE)
    # The original page was 32.5 MB. Retain at least the planned 80% reduction.
    assert (root / "examples/generating_graphs_by_proximity.html").stat().st_size < 6_500_000
    assert not (root / "overrides").exists()
    assert not (root / "hooks.py").exists()
    assert not (root / "examples/data").exists()
    assert not list(root.rglob("*.psd"))
    assert not list(root.rglob("*.pyc"))
    assert not list(root.rglob(".DS_Store"))


@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("width", [390, 1440])
def test_responsive_notebook_layout(
    browser: Browser, docs_server: str, width: int, dark: bool
) -> None:
    """Render notebook plots and keep navigation usable in both palettes."""
    context = browser.new_context(
        viewport={"width": width, "height": 900},
        color_scheme="dark" if dark else "light",
        reduced_motion="reduce",
    )
    context.add_init_script("window.__city2graphAnalytics = true")
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(docs_server + "examples/add_metapaths.html", wait_until="domcontentloaded")
    page.locator(".jupyter-wrapper").wait_for()
    page.wait_for_function("document.querySelector('.md-content img').naturalWidth > 0")
    scheme = page.locator("body").get_attribute("data-md-color-scheme")
    assert scheme == ("city2graph-dark" if dark else "city2graph-light")
    assert page.locator('.jupyter-wrapper img[src^="data:image/png"]').count() == 0
    assert page.locator('.jupyter-wrapper script[type="text/x-mathjax-config"]').count() == 0
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
    if width < 1000:
        page.locator('.md-header label[for="__drawer"]').click()
        assert page.locator("#__drawer").is_checked()
    assert not errors
    context.close()


@pytest.mark.parametrize("navigate_while_loading", [False, True])
def test_maths_after_delayed_load_and_instant_navigation(
    page: Page,
    docs_server: str,
    navigate_while_loading: bool,
) -> None:
    """Render real TeX after startup, including a page replaced during loading."""
    assert SITE
    html = Path(SITE, "index.html").read_text()
    from bs4 import BeautifulSoup  # noqa: PLC0415

    soup = BeautifulSoup(html, "html.parser")
    article = soup.select_one(".md-content__inner")
    assert article is not None
    article.clear()
    article.append(
        BeautifulSoup(
            '<h1>Maths smoke test</h1><span class="arithmatex">\\(x^2 + y^2\\)</span>',
            "html.parser",
        )
    )
    page.route(
        "**/latest/maths-smoke.html",
        lambda route: route.fulfill(
            content_type="text/html",
            body=str(soup),
        ),
    )
    errors: list[str] = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    requests: list[str] = []
    page.on("request", lambda request: requests.append(request.url))
    if navigate_while_loading:
        page.add_init_script("""(() => {
          const append = Element.prototype.append;
          Element.prototype.append = function(...nodes) {
            if (nodes.some(n => n.tagName === 'SCRIPT' && n.src.includes('mathjax@'))) {
              setTimeout(() => append.apply(this, nodes), 2000);
              return;
            }
            return append.apply(this, nodes);
          };
        })()""")
    page.goto(docs_server + "maths-smoke.html", wait_until="domcontentloaded")
    page.wait_for_function(
        "performance.getEntriesByType('resource').some(entry => entry.name.endsWith('/sitemap.xml'))"
    )
    page.evaluate("window.docsNavigationSentinel = true")
    if not navigate_while_loading:
        page.locator("mjx-container").wait_for(timeout=45000)
    page.locator('.md-tabs a[href$="installation.html"]').click()
    page.get_by_role("heading", name="Install City2Graph", exact=False).first.wait_for()
    page.go_back()
    page.locator("mjx-container").wait_for(timeout=45000)
    assert page.evaluate("window.docsNavigationSentinel") is True
    assert len([url for url in requests if url.endswith("mathjax@3.2.2/es5/tex-mml-chtml.js")]) == 1
    assert not errors
