"""Sphinx configuration for the MedicalImageAnalysis documentation site."""

from __future__ import annotations

from datetime import date
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "MedicalImageAnalysis"
author = "Caleb OConnor"
copyright = f"{date.today().year}, {author}"

try:
    release = pkg_version("medicalimageanalysis")
except PackageNotFoundError:
    release = "development"

version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "paper"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autosummary_generate = True
autosummary_imported_members = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "show-inheritance": True,
    "undoc-members": False,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True

myst_enable_extensions = ["colon_fence", "deflist"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# The package depends on several heavy scientific libraries that may not be
# installed in lightweight documentation environments.
autodoc_mock_imports = [
    "ants",
    "blosc2",
    "cryptography",
    "cv2",
    "fast_simplification",
    "gdcm",
    "numba",
    "open3d",
    "pandas",
    "PIL",
    "psutil",
    "pyacvd",
    "pydicom",
    "pylibjpeg",
    "pylibjpeg_libjpeg",
    "pymeshfix",
    "pytetwild",
    "scipy",
    "shapely",
    "SimpleITK",
    "skimage",
    "vtk",
    "pyvista",
]

html_theme = "pydata_sphinx_theme"
html_title = project
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "navigation_with_keys": True,
    "show_toc_level": 2,
    "navbar_align": "content",
    "header_links_before_dropdown": 6,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "use_edit_page_button": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/caleb-oconnor/MedicalImageAnalysis",
            "icon": "fa-brands fa-github",
        }
    ],
    "secondary_sidebar_items": ["page-toc"],
    "announcement": "MedicalImageAnalysis documentation site.",
}

html_context = {
    "github_user": "caleb-oconnor",
    "github_repo": "MedicalImageAnalysis",
    "github_version": "main",
    "doc_path": "docs",
}
