# Configuration file for the Sphinx documentation builder.

import os
import sys
import subprocess
from datetime import datetime

try:
    from zhdate import ZhDate
    ZHDATE_AVAILABLE = True
except ImportError:
    ZHDATE_AVAILABLE = False

sys.path.insert(0, os.path.abspath(".."))

def get_ganzhi_year(zh_date):
    chinese = zh_date.chinese()
    parts = chinese.split()
    for part in parts:
        if '年' in part and len(part) == 3:
            return part
    return None

def get_git_last_updated(file_path):
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ci", "--", file_path],
            cwd=os.path.abspath(".."),
            capture_output=True,
            text=True,
            check=True
        )
        if result.stdout.strip():
            git_date = datetime.strptime(result.stdout.strip(), "%Y-%m-%d %H:%M:%S %z")
            return git_date.strftime("%Y-%m-%d")
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        pass
    return None

from importlib.metadata import version as _version

try:
    release = _version("seispolarity")
except Exception:
    release = "dev"

project = "SeisPolarity"
now = datetime.now()
lunar_year = None
if ZHDATE_AVAILABLE:
    try:
        zh_date = ZhDate.from_datetime(now)
        ganzhi = get_ganzhi_year(zh_date)
        if ganzhi:
            lunar_year = ganzhi
    except Exception:
        pass

copyright = f"{now.year}, He XingChen {lunar_year or ''}".strip()
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

html_last_updated_fmt = "%Y-%m-%d"

def get_lunar_date(gregorian_date):
    if not ZHDATE_AVAILABLE:
        return None
    try:
        zh_date = ZhDate.from_datetime(gregorian_date)
        ganzhi_year = get_ganzhi_year(zh_date)
        chinese_str = zh_date.chinese()
        parts = chinese_str.split()
        if ganzhi_year and len(parts) >= 1:
            month_day_str = parts[0]
            if '腊月' in month_day_str or '正月' in month_day_str or '二月' in month_day_str:
                month_day = month_day_str.replace('二零二五年', '').replace('二零二六年', '')
                if month_day:
                    return f"{ganzhi_year}{month_day}"
            else:
                for part in parts:
                    if '月' in part and '年' not in part:
                        return f"{ganzhi_year}{part}"
        return None
    except Exception:
        return None

def setup(app):
    app.add_config_value('get_lunar_date', get_lunar_date, 'html')
    
    def html_page_context_handler(app, pagename, templatename, context, doctree):
        source_suffix = app.config.source_suffix
        if not isinstance(source_suffix, list):
            source_suffix = list(source_suffix.keys())
        
        source_file = None
        for suffix in source_suffix:
            possible_path = os.path.join(app.srcdir, pagename + suffix)
            if os.path.exists(possible_path):
                source_file = possible_path
                break
        
        if source_file:
            last_updated_str = get_git_last_updated(source_file)
            if last_updated_str:
                context['last_updated'] = last_updated_str
                try:
                    last_updated_dt = datetime.strptime(last_updated_str, "%Y-%m-%d")
                    lunar_date_str = get_lunar_date(last_updated_dt)
                    if lunar_date_str:
                        context['lunar_last_updated'] = lunar_date_str
                except Exception:
                    pass
    
    app.connect('html-page-context', html_page_context_handler)

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
