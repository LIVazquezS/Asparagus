# Configuration file for the Sphinx documentation builder.
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join('.','..', '..')))
# sys.path.insert(1,os.path.abspath(os.path.join('.','..', '..')))
# -- Project information

project = 'Asparagus bundle'
copyright = '2023, L.I.Vazquez-Salazar, K. Toepfer & M. Meuwly'
author = 'L.I.Vazquez-Salazar & K. Toepfer'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'myst_parser',]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}

autodoc_mock_imports = ['ase', 'torch', 'torch-ema', 'numpy', 'tensorboard', 'xtb','h5py','pandas','matplotlib','seaborn','scipy','pytest','.src']

intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'