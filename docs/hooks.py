"""Optimise generated documentation without changing notebook source files."""

import ast
import base64
import binascii
import hashlib
import io
import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from bs4 import BeautifulSoup
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.plugins import event_priority
from mkdocs.structure.pages import Page
from mkdocs.structure.files import Files
from PIL import Image, UnidentifiedImageError


@event_priority(100)  # type: ignore[untyped-decorator]
def on_files(files: Files, config: MkDocsConfig) -> Files:
    """Show Git revision dates only for pages present in the current commit.

    New local pages have no revision date. Excluding them also prevents the
    date plugin from computing inconsistent first/last build-time fallbacks.
    """
    plugin = config.plugins.get("git-revision-date-localized")
    git = shutil.which("git")
    if plugin is None or git is None:
        return files
    result = subprocess.run(
        [git, "-C", config.docs_dir, "ls-tree", "-r", "--name-only", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        return files
    committed = set(result.stdout.splitlines())
    excluded = set(plugin.config["exclude"])
    excluded.update(file.src_uri for file in files if file.is_documentation_page() and file.src_uri not in committed)
    plugin.config["exclude"] = sorted(excluded)
    return files


def on_page_markdown(markdown: str, **_kwargs: object) -> str:
    """Render GTFS default tables from the registry used by the generator."""
    if "<!-- gtfs-defaults:" not in markdown:
        return markdown
    registry = json.loads(
        (Path(__file__).parents[1] / "city2graph/gtfs/defaults.json").read_text(encoding="utf-8")
    )
    values = {field: entry["value"] for field, entry in registry["settings"].items()}
    resolution = [
        "| Condition | Your input | Source value, if available | Default |",
        "|---|---|---|---|",
        "| Interval | `services.headway_secs`, then `headway_secs` or `group_headway_secs` | "
        "Interpreted base and conditional intervals | "
        f"{values['group_headway_secs']} s shared across a route-direction group |",
        "| Departure window | Service-row boundary, then `start_time` / `end_time` | "
        f"Interpreted departure periods | {values['start_time']}–{values['end_time']} |",
        "| Leg time | Complete offsets or leg time, then per-leg or global speed | "
        "Interpreted route duration | Distance / [mode speed](#default-mode-speeds) |",
        "| Additional dwell | Included in complete offsets; otherwise per-stop or global dwell | "
        f"No source field used | {values['dwell_time_secs']} s |",
        "| Active dates | `service_date`, or `calendar` / `calendar_dates` | "
        "Source weekday restrictions apply within these dates | "
        f"{values['service_days']} reference days from the {values['service_date']} |",
        "| Agency metadata | `agency` table | Interpreted operator, URL and timezone | "
        "Geographical timezone inference; placeholders for unresolved fields |",
    ]
    controls = [
        "| Parameter | What it sets | If omitted |",
        "|---|---|---|",
        "| `headway_secs` (s) | Interval for each pattern | Source interval, then group allocation |",
        "| `group_headway_secs` (s) | Shared interval for a route-direction group | "
        f"Source intervals, otherwise a shared {values['group_headway_secs']} s interval |",
        "| `start_time` / `end_time` | First-stop departure window | Source hours, otherwise "
        f"{values['start_time']}–{values['end_time']} |",
        "| `speed_kph` (km/h) | Speed for untimed legs | Usable source duration, otherwise "
        f"mode speed, e.g. {registry['speeds_kph']['1']} km/h for subway |",
        "| `dwell_time_secs` (s) | Additional dwell at each stop | "
        f"{values['dwell_time_secs']} s |",
        "| `distance_method` | `auto`, `shape` or `geodesic` distance estimation | "
        f"`{values['distance_method']}` |",
        "| `service_date` | Single-day synthetic timetable (`YYYYMMDD`); determines the weekday | Supplied calendar, otherwise "
        f"{values['service_days']} reference days from the {values['service_date']} |",
        "| `calendar` / `calendar_dates` | Weekdays, date ranges and exceptions | "
        "A calendar for the selected date or reference week |",
        "| `services` | Per-pattern intervals, windows or fixed departures | "
        "Complete all selected patterns from source/default values |",
        "| `agency` | Operator name, URL and timezone | Source metadata, then geographical "
        "timezone inference and placeholder metadata |",
        "| `optional_tables` | Additional GTFS tables | None added |",
    ]
    speeds = [
        "| Mode (`route_type`) | Average speed (km/h) | Extended route types |",
        "|---|---:|---|",
    ]
    speeds.extend(
        f"| {registry['mode_names'][mode]} ({mode}) | {speed} | "
        + ", ".join(str(code) for code in registry["extended_route_types"].get(mode, []))
        + " |"
        for mode, speed in registry["speeds_kph"].items()
    )
    setting_groups = {
        "Operating assumptions": (
            "group_headway_secs", "start_time", "end_time", "dwell_time_secs",
            "exact_times", "distance_method",
        ),
        "Reference dates": ("service_date", "service_days", "weekday_policy"),
        "Agency metadata": ("agency_name", "agency_url", "agency_timezone"),
        "OSM retrieval": ("use_source_tags", "endpoint", "timeout", "overwrite_source"),
        "Saving and loading": ("include_metadata", "overwrite"),
    }
    for title, fields in setting_groups.items():
        settings = [
            "| Setting | Default and unit | Scope and override |",
            "|---|---|---|",
        ]
        for field in fields:
            entry = registry["settings"][field]
            value = json.dumps(entry["value"]) if isinstance(entry["value"], bool) else str(entry["value"])
            settings.append(
                f"| `{field}` | `{value}` ({entry['unit']}) | {entry['scope']}. "
                f"Override: {entry['override']}. |"
            )
        slug = title.lower().replace(" ", "-")
        markdown = markdown.replace(f"<!-- gtfs-defaults:{slug} -->", "\n".join(settings))
    return (
        markdown.replace(
            "    <!-- gtfs-defaults:controls -->",
            "\n".join(f"    {line}" for line in controls),
        )
        .replace("<!-- gtfs-defaults:controls -->", "\n".join(controls))
        .replace("<!-- gtfs-defaults:resolution -->", "\n".join(resolution))
        .replace("<!-- gtfs-defaults:speeds -->", "\n".join(speeds))
    )


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
