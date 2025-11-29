# Configuration file for the Sphinx documentation builder.
#
# This file configures Sphinx documentation generation for the city2graph package.
# It includes settings for autodoc, cross-references (intersphinx), SEO optimization,
# and theme customization for the PyData Sphinx Theme.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
# Add the project root to Python path so Sphinx can import the package
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import pypandoc

# Configure nbsphinx to use the pandoc binary from pypandoc
nbsphinx_pandoc_path = pypandoc.get_pandoc_path()
# Also add to PATH as fallback
os.environ["PATH"] = os.path.dirname(nbsphinx_pandoc_path) + os.pathsep + os.environ["PATH"]

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "city2graph"
copyright = "2025, Yuta Sato & city2graph developers"
author = "Yuta Sato"
release = "0.1.7"
version = "0.1.7"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Extensions for enhanced documentation generation
extensions = [
    "sphinx.ext.autodoc",          # Automatic documentation from docstrings
    "sphinx.ext.autosummary",      # Generate summary tables for modules/classes
    "sphinx.ext.napoleon",         # Support for NumPy and Google style docstrings
    "sphinx.ext.viewcode",         # Add source code links to documentation
    "sphinx.ext.mathjax",          # Math support via MathJax
    "sphinx.ext.intersphinx",      # Cross-references to other projects
    "sphinx_autodoc_typehints",    # Type hints in documentation
    "sphinxext.opengraph",         # Open Graph meta tags for social sharing
    "sphinx_plotly_directive",     # Plotly figure support
    "nbsphinx",                    # Jupyter notebook support
    "sphinx_sitemap",              # Generate sitemap.xml for SEO
]

# Open Graph (social media) configuration
ogp_site_url = "https://city2graph.net/"
ogp_image = "https://city2graph.net/_static/social_preview_city2graph.png"
ogp_description = "Transform urban data into graphs for spatial analysis and Graph Neural Networks"
ogp_type = "website"

# Template and exclusion settings
templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# -- SEO and metadata configuration -----------------------------------------
# Optimize documentation for search engines and social media sharing

# Title for the HTML documentation
html_title = "GeoAI with Graph Neural Network (GNN) in Python"
# Base URL for sitemap and canonical URLs
html_baseurl = "https://city2graph.net/"
# Include robots.txt in the output
html_extra_path = ["robots.txt"]
# Default meta description for pages
html_meta = {
    "application-name": "city2graph",
    "description": "A Python library for converting geospatial datasets into graph representations, providing an integrated interface for `GeoPandas <https://geopandas.org/>`_, `NetworkX <https://networkx.org/>`_, and `Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/>`_ across multiple domains (e.g. streets, transportations, OD matrices, POI proximities, etc.). It enables researchers and practitioners to seamlessly develop advanced GeoAI and geographic data science applications. For more information, please reach out to the document (https://city2graph.net).",
    "author": "Yuta Sato",
}
# Sitemap settings for better SEO
sitemap_url_scheme = "{link}"
sitemap_locales = ["en"]

# Documentation language
language = "en"

# -- Options for autodoc -----------------------------------------------------
# Configure automatic documentation generation from docstrings

autodoc_member_order = "bysource"        # Order members as they appear in source
autodoc_typehints = "description"        # Include type hints in parameter descriptions
autoclass_content = "both"               # Include both class and __init__ docstrings
add_module_names = False                 # Don't show module names in function signatures

# -- Options for autosummary ---------------------------------------------
# Configure automatic summary table generation

# -- Options for napoleon (NumPy/Google docstring parsing) --------------
# Configure parsing of NumPy and Google style docstrings

# -- Options for type hints ----------------------------------------------
# Configure display of type hints in documentation

# -- Options for numpydoc -----------------------------------------------
# Legacy NumPy docstring extension (kept for compatibility)

numpydoc_show_class_members = False      # Don't show class members in class docs
numpydoc_show_inherited_class_members = False # Don't show inherited members
numpydoc_class_members_toctree = False   # Don't create toctree for class members

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Theme and static files configuration
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

# Set the favicon for the browser tab
html_favicon = "_static/city2graph_logo.png"

# Custom CSS files
html_css_files = [
    "custom.css",
]

# Custom JavaScript files (Google Analytics)
html_js_files = [
    ("https://www.googletagmanager.com/gtag/js?id=G-YLJ6HSS8C4", {"async": "async"}),
    "gtag.js",
]

# Theme-specific configuration
html_theme_options = {
    # GitHub integration
    "github_url": "https://github.com/c2g-dev/city2graph",
    "use_edit_page_button": False,

    # Navigation and layout
    "show_toc_level": 2,                 # Show 2 levels in table of contents
    "navigation_with_keys": True,        # Enable keyboard navigation
    "show_nav_level": 1,                 # Show 1 level in navigation
    "show_breadcrumbs": True,            # Enable breadcrumbs
    "navbar_start": ["navbar-logo"],     # Left side of navbar
    "navbar_center": ["navbar-nav"],     # Center of navbar
    "navbar_end": ["navbar-icon-links", "theme-switcher"], # Right side of navbar

    # Logo configuration
    "logo": {
        "image_light": "_static/city2graph_logo_wide.png",
        "image_dark": "_static/city2graph_logo_wide_dark.png",
        "alt_text": "city2graph - GeoAI with Graph Neural Network (GNN) in Python",
    },

    # Footer configuration
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],

    # Search configuration
    "search_bar_text": "Search city2graph documentation...",

    # Social media and SEO
    "open_graph_image": "_static/city2graph_logo_wide.png",
    "open_graph_description": "A Python package converting geospatial data into graphs for spatial network analysis and Graph Neural Networks",
    "thumbnail": "https://city2graph.net/_static/social_preview_city2graph.png",

    # Page layout options
    "collapse_navigation": False,        # Keep navigation expanded
    "sticky_navigation": True,           # Make navigation sticky
    "includehidden": True,               # Include hidden toctree entries
    "titles_only": False,                # Show full titles, not just page names
    "default_mode": "auto",              # Options: "auto", "light", "dark"
}

# Additional HTML context variables
html_context = {
    "github_user": "c2g-dev",
    "github_repo": "city2graph",
    "github_version": "main",
    "doc_path": "docs/source",
}

# HTML output options
html_copy_source = True                  # Copy source files to output
html_show_sourcelink = True             # Show source links
html_show_sphinx = True                 # Show "Created with Sphinx" footer
html_show_copyright = True              # Show copyright notice
html_last_updated_fmt = "%b %d, %Y"     # Format for last updated date

# Custom sidebar templates
html_sidebars = {
    "**": ["sidebar-nav-bs"],
}

# -- Intersphinx configuration -----------------------------------------------
# Configure cross-references to external documentation

intersphinx_mapping = {
    # Python standard library
    "python": ("https://docs.python.org/3", None),

    # Scientific computing stack
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),

    # Geospatial libraries
    "geopandas": ("https://geopandas.org/en/stable/", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable/", None),
    "fiona": ("https://fiona.readthedocs.io/en/latest/", None),
    "pyproj": ("https://pyproj4.github.io/pyproj/stable/", None),

    # Graph libraries
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "rustworkx": ("https://www.rustworkx.org/", None),

    # Deep learning and PyTorch ecosystem
    "torch": ("https://pytorch.org/docs/stable/", None),
    "torch_geometric": ("https://pytorch-geometric.readthedocs.io/en/latest/", None),

    # Urban analysis libraries
    "momepy": ("http://docs.momepy.org/en/stable/", None),
    "osmnx": ("https://osmnx.readthedocs.io/en/stable/", None),
    "libpysal": ("https://pysal.org/libpysal/", None),
}

# Intersphinx timeout in seconds
intersphinx_timeout = 10

# -- Additional Sphinx configuration -------------------------------------

# Source file extensions
source_suffix = {
    ".rst": None,
    ".md": "myst_parser",
}

# Master document (main entry point)
master_doc = "index"

# Suppress certain warnings
suppress_warnings = [
    "image.nonlocal_uri",      # Suppress warnings about external images
    "toc.secnum",              # Suppress section numbering warnings
]

# Add any paths that contain custom static files (such as style sheets)
html_static_path = ["_static"]

# Add any paths that contain templates
templates_path = ["_templates"]
