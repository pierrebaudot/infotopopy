 
.. _Installation:

Installation
============

Requirements
++++++++++++

Infotopo relies on two packages :

* `NumPy <https://www.numpy.org/>`_
* `Networkx <https://networkx.github.io/documentation/stable/auto_examples/index.html>`_

Then if you want to be able to run the examples you'll need to have :

* `Matplotlib <https://matplotlib.org/>`_
* `Seaborn <https://seaborn.pydata.org/>`_
* `Pandas <https://pandas.pydata.org/>`_
* `Scikit learn <https://scikit-learn.org/stable/>`_


Standard installation
+++++++++++++++++++++

Infotopo can be installed using pip. In a terminal, run the following command :

.. code-block:: shell

    pip install infotopo

And if you want want to update to the latest version :

.. code-block:: shell

    pip install -U infotopo

Install the most up-to-date version
+++++++++++++++++++++++++++++++++++

The latest version is hosted on `github <https://github.com/pierrebaudot/infotopopy>`_.
This is always going to be the most up-to-date version, with the latest features and fixes.
If you want to install this version, open a terminal and run the following commands :

.. code-block:: shell

    git clone https://github.com/pierrebaudot/infotopopy.git
    cd infotopo/
    python setup.py develop

Finally, if you want to update your version you can use :

.. code-block:: shell

    git pull
