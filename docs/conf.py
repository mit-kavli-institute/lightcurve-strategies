"""Sphinx configuration for lightcurve-strategies."""

from importlib.metadata import version as _get_version

project = "lightcurve-strategies"
author = ""
release = _get_version("lightcurve-strategies")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "matplotlib.sphinxext.plot_directive",
]

# -- Plot directive settings -------------------------------------------------
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = [("png", 150)]

html_theme = "furo"
html_static_path: list[str] = []
templates_path = ["_templates"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "hypothesis": ("https://hypothesis.readthedocs.io/en/latest/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
