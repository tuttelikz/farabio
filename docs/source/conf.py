# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import sphinx_rtd_theme
import shutil
from recommonmark.parser import CommonMarkParser

# sys.path.insert(0, os.path.abspath('../../farabio/models/'))  # models added here
sys.path.insert(0, os.path.abspath('../../farabio/'))  # models added here
sys.path.insert(0, os.path.abspath('../../'))  # models added here

# -- Project information -----------------------------------------------------

project = 'farabio'
copyright = '2021, MIT License'
author = 'San Askaruly'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

# add following to the extensions
# for numpy and google style:
# 'sphinx.ext.napoleon'

#graphviz_dot = shutil.which('dot')

extensions = [
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx_rtd_theme",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx_git",
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'matplotlib.sphinxext.plot_directive',
    'recommonmark',
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
    'sphinx.ext.autosectionlabel'
]

autodoc_default_options = {
    'members': 'var1, var2',
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [

]

source_parsers = {
    '.md': CommonMarkParser,
}
source_suffix = ['.rst', '.md']
autosectionlabel_prefix_document = True

numpydoc_show_class_members = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []
