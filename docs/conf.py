# Configuration file for the Sphinx documentation builder.

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(".."))

from importlib.metadata import version as _version

try:
    release = _version("seispolarity")
except Exception:
    release = "dev"

project = "SeisPolarity"
copyright = f"{datetime.now().year}, He XingChen"
author = "He XingChen"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx_rtd_theme",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

add_module_names = False

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/seispolarity_logo.svg"
html_favicon = "_static/seispolarity_logo.svg"
html_title = "SeisPolarity"

html_theme_options = {
    "logo_only": True,
    "display_version": True,
}

html_css_files = [
    "custom.css",
]

html_js_files = [
    "custom.js",
]

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__"
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

myst_heading_anchors = 3
myst_fence_as_directive = ["mermaid"]
myst_enable_checkboxes = True
myst_url_schemes = ("http", "https", "mailto", "ftp")

myst_number_code_blocks = ["(python)", "python"]
myst_highlight_language_blocks = ["default", "python", "bash"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
