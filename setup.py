#!/usr/bin/env python
# coding: utf-8
import os
import setuptools

__version__ = "0.2.4"
NAME = 'Infotopo'
AUTHOR = "Pierre Baudot"
MAINTAINER = "Pierre Baudot"
EMAIL = 'pierre.baudot@gmail.com'
KEYWORDS = "topological entropy supervised learning"
DESCRIPTION = ("InfoTopo: Topological Information Data Analysis. Deep "
               "statistical unsupervised and supervised learning.")
URL = 'https://github.com/pierrebaudot/infotopopy'
DOWNLOAD_URL = ("https://github.com/pierrebaudot/infotopopy/archive/v" +
                __version__ + ".tar.gz")
# Data path :
PACKAGE_DATA = {}

def read(fname):
    """Read README and LICENSE."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setuptools.setup(
    name=NAME,
    version=__version__,
    author=AUTHOR,
    maintainer=MAINTAINER,
    author_email=EMAIL,
    url=URL,
    download_url=DOWNLOAD_URL,
    packages=setuptools.find_packages(),
    package_dir={'infotopo': 'infotopo'},
    package_data=PACKAGE_DATA,
    include_package_data=True,
    license='BSD 3-Clause "New" or "Revised" License',
    description=DESCRIPTION,
    long_description=read('README.md'),
    platforms='any',
    setup_requires=['numpy'],
    install_requires=[
        "numpy", "networkx", "matplotlib"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)