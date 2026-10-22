"""Optimise generated documentation without changing notebook source files."""

import ast
import base64
import binascii
import hashlib
import io
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from bs4 import BeautifulSoup
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.plugins import event_priority
from mkdocs.structure.pages import Page
from PIL import Image, UnidentifiedImageError


def _asset(data: bytes, suffix: str, config: MkDocsConfig) -> str:
    name = f"assets/generated/{hashlib.sha256(data).hexdigest()}{suffix}"
    target = Path(config.site_dir, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(data)
    return name


def _relative_url(asset: str, page: Page) -> str:
    return "../" * len(Path(page.file.dest_uri).parent.parts) + asset


@event_priority(100)  # type: ignore[untyped-decorator]
def on_post_page(output: str, page: Page, config: MkDocsConfig) -> str:
    """Externalise notebook assets and reserve space for local images."""
    soup = BeautifulSoup(output, "html.parser")
    article = soup.select_one(".md-content__inner")
    if article is None:
        return output

    if article.select_one(".jupyter-wrapper"):
        # Saved widget state can include megabytes of obsolete map tiles even
        # when the notebook has no widget views. Keep state for live views.
        if not article.select_one(
            'script[type="application/vnd.jupyter.widget-view+json"]'
        ):
            for state in article.select(
                'script[type="application/vnd.jupyter.widget-state+json"]'
            ):
                state.decompose()
        # nbconvert emits MathJax 2 configuration even when no URL is set.
        # The site's loader owns MathJax 4 and handles instant navigation.
        for script in article.find_all("script"):
            if script.get("type") == "text/x-mathjax-config" or (
                "src" in script.attrs and not script.get("src")
            ):
                script.decompose()

        for style in article.find_all("style"):
            css = style.get_text()
            # mkdocs-jupyter appends a Python representation of the Pygments
            # stylesheet list after having already emitted the actual CSS.
            if css.lstrip().startswith("["):
                try:
                    duplicate = ast.literal_eval(css.strip())
                except (SyntaxError, ValueError):
                    duplicate = None
                if isinstance(duplicate, list) and all(
                    isinstance(x, str) for x in duplicate
                ):
                    style.decompose()
                    continue
            if len(css) > 1024:
                link = soup.new_tag("link", rel="stylesheet")
                link["href"] = _relative_url(_asset(css.encode(), ".css", config), page)
                style.replace_with(link)

    first_content_image = True
    for image in soup.find_all("img"):
        src = image.get("src", "")
        if not isinstance(src, str):
            continue
        data = None
        if src.startswith("data:image/png;base64,"):
            try:
                data = base64.b64decode(src.partition(",")[2], validate=True)
            except (ValueError, binascii.Error):
                continue
            image["src"] = _relative_url(_asset(data, ".png", config), page)
        else:
            url = urlsplit(src)
            if url.scheme or url.netloc or not url.path or url.path.startswith("/"):
                continue
            source = (Path(page.file.abs_src_path).parent / unquote(url.path)).resolve()
            if (
                source.is_relative_to(Path(config.docs_dir).resolve())
                and source.is_file()
            ):
                data = source.read_bytes()

        if data:
            try:
                with Image.open(io.BytesIO(data)) as raster:
                    # Preserve explicit display sizes such as the Crossmark icon.
                    if "width" not in image.attrs and "height" not in image.attrs:
                        image["width"], image["height"] = map(str, raster.size)
                    elif "width" in image.attrs and "height" not in image.attrs:
                        width = str(image["width"])
                        if width.isdigit():
                            image["height"] = str(
                                round(int(width) * raster.height / raster.width)
                            )
                    elif "height" in image.attrs and "width" not in image.attrs:
                        height = str(image["height"])
                        if height.isdigit():
                            image["width"] = str(
                                round(int(height) * raster.width / raster.height)
                            )
            except (UnidentifiedImageError, OSError):
                pass
        if article in image.parents:
            if image.get("alt") in {
                None,
                "No description has been provided for this image",
            }:
                heading = image.find_previous(re.compile(r"^h[1-6]$"))
                image["alt"] = "Plot: " + (
                    heading.get_text(" ", strip=True).rstrip("¶").strip()
                    if heading
                    else page.title
                )
            if first_content_image:
                image["loading"] = "eager"
                if page.is_homepage:
                    image["fetchpriority"] = "high"
                first_content_image = False
            else:
                image["loading"] = "lazy"
            image["decoding"] = "async"

    for video in article.find_all("video"):
        video["preload"] = "none"

    return str(soup)


class _References(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.references: list[str] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        identifier = values.get("id") or (values.get("name") if tag == "a" else None)
        if identifier:
            self.ids.add(identifier)
        attributes = {
            "a": ("href",),
            "img": ("src",),
            "script": ("src",),
            "link": ("href",),
            "iframe": ("src",),
            "source": ("src",),
            "video": ("src", "poster"),
            "audio": ("src",),
        }
        for attribute in attributes.get(tag, ()):
            if value := values.get(attribute):
                self.references.append(value)
        if tag in {"img", "source"} and (srcset := values.get("srcset")):
            if not srcset.startswith("data:"):
                self.references.extend(
                    item.strip().split()[0]
                    for item in srcset.split(",")
                    if item.strip()
                )


@event_priority(-100)  # type: ignore[untyped-decorator]
def on_post_build(config: MkDocsConfig) -> None:
    """Fail builds with missing internal pages, anchors, or static assets."""
    root = Path(config.site_dir).resolve()
    # MkDocs deliberately preserves dotfiles when cleaning an existing build.
    for metadata in root.rglob(".DS_Store"):
        metadata.unlink()
    site_url = urlsplit(config.site_url)
    prefix = site_url.path.rstrip("/") + "/"
    documents: dict[Path, _References] = {}
    for path in root.rglob("*.html"):
        parser = _References()
        parser.feed(path.read_text(encoding="utf-8"))
        documents[path] = parser

    references = [
        (path, ref) for path, doc in documents.items() for ref in doc.references
    ]
    for path in root.rglob("*.css"):
        css = path.read_text(encoding="utf-8")
        references.extend(
            (path, match.strip(" \"'")) for match in re.findall(r"url\(([^)]+)\)", css)
        )

    errors = set()
    for source, reference in references:
        url = urlsplit(reference)
        if url.scheme and url.scheme not in {"http", "https"}:
            continue
        if url.netloc and url.netloc != site_url.netloc:
            continue
        url_path = unquote(url.path)
        if url_path.startswith("/") or url.netloc:
            # Other published versions are outside this build's output tree.
            if not url_path.startswith(prefix):
                continue
            target = root / url_path.removeprefix(prefix)
        else:
            target = source.parent / url_path if url_path else source
        target = target.resolve()
        if not target.is_relative_to(root):
            errors.add(f"{source.relative_to(root)}: outside site: {reference}")
            continue
        if target.is_dir():
            target /= "index.html"
        if not target.is_file():
            errors.add(f"{source.relative_to(root)}: missing file: {reference}")
        elif url.fragment and target in documents:
            if unquote(url.fragment) not in documents[target].ids:
                errors.add(f"{source.relative_to(root)}: missing anchor: {reference}")
    if errors:
        raise RuntimeError(
            "Broken documentation references:\n" + "\n".join(sorted(errors))
        )
