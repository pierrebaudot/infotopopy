#!/usr/bin/env python
# coding: utf-8


import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="infotopo", # Replace with your own username
    version="0.2.9",
    license='BSD 3-Clause "New" or "Revised" License',
    author="Pierre Baudot",   
    author_email="pierre.baudot@gmail.com",
    description="InfoTopo: Topological Information Data Analysis. Deep statistical unsupervised and supervised learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pierrebaudot/infotopopy/archive/0.2.9.tar.gz",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)