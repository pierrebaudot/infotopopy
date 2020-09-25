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
import sphinx_bootstrap_theme
import infotopo
from sphinx_gallery.sorting import FileNameSortKey, ExplicitOrder


# -- Project information -----------------------------------------------------

project = 'infotopo'
copyright = '2020, Pierre Baudot'
author = 'Pierre Baudot'

# The full version, including alpha/beta/rc tags
version = infotopo.__version__
release = infotopo.__version__
sys.path.insert(0, os.path.abspath('sphinxext'))

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx_gallery.gen_gallery',
    'sphinxcontrib.bibtex',
    'numpydoc',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()
html_theme = 'bootstrap'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme_options = {
    'bootstrap_version': "3",
    'navbar_sidebarrel': False,
    'navbar_pagenav': False,
    'navbar_pagenav_name': "Page",
    'globaltoc_depth': -1,
    'globaltoc_includehidden': "true",
    'source_link_position': "nav",
    'navbar_class': "navbar",
    'bootswatch_theme': "readable",
    'navbar_fixed_top': True,
    'navbar_links': [
                        ("Method", "method"),
                        ("Tutorial", "tutorial"),
                        ("Install", "install"),
                        ("API", "api"),
                        ("Examples", "auto_examples/index")
    ],
}

sphinx_gallery_conf = {
    'examples_dirs': '../../examples',
    'gallery_dirs': 'auto_examples',
    'backreferences_dir': 'generated',
}

autosummary_generate = True
autodoc_member_order = 'groupwise'
autodoc_default_flags = ['members', 'inherited-members', 'no-undoc-members']