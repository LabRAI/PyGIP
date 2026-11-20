# docs/source/conf.py
import os
import sys

sys.path.insert(0, os.path.abspath('../../'))

# Import scipy.stats before autodoc mocks torch: newer SciPy probes
# torch.Tensor at import time and fails on the mocked module.
import scipy.stats  # noqa: E402,F401

project = 'PyGIP'
copyright = '2025, RAILab'
author = 'RAILab'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode',
              'sphinx.ext.autosummary',
              'sphinx_autodoc_typehints'
              ]

templates_path = ['_templates']
exclude_patterns = []
autosummary_generate = True

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': True,
    'show-inheritance': True,
}

autodoc_mock_imports = [
    "torch", "torchvision", "torchaudio",
    "dgl",
    "torch_geometric", "pyg_lib",
    "torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv",
    "pylibcugraphops", "pandas"
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_baseurl = "https://labrai.github.io/PyGIP/"

html_theme_options = {
    "navigation_with_keys": True,
    "sidebar_hide_name": False,
}
