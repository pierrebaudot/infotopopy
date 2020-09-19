Topological Learning
====================

The presentation of the basic methods and principles we made so far mostly relied on basic information lattice decomposition and simplex structure.
In what follows, we will go one stepp further by introducing to simplicial complexes of information which can display much richer structures. This will be the 
occasion to study more in depth information paths, the analog of homotopical paths in information theory. 
Poincaré Shannon Machine . Information function as loss function (Boltzmann - Helmoltz machine, maxent Jaynes infomax Linsker nadal parga Sejnowsky ... untill current cross entropy
and focal loss = deformed probability ).  


We will illustrate the computation of free energy complex (or  :math:`I_k` complex) on the synthetic dataset 

.. code:: python3
        import pandas as pd
        dataset = pd.read_csv(r"/home/pierre/Documents/Data/lucas0_train.csv")  # csv to download at http://www.causality.inf.ethz.ch/data/LUCAS.html
        dataset_df = pd.DataFrame(dataset, columns = dataset.columns)
        dataset = dataset.to_numpy()
        dimension_max = dataset.shape[1]
        dimension_tot = dataset.shape[1]
        sample_size = dataset.shape[0]
        nb_of_values = 2
        forward_computation_mode = False
        work_on_transpose = False
        supervised_mode = False
        sampling_mode = 1
        deformed_probability_mode = False 


Unsupervised
------------

Information Complexes
~~~~~~~~~~~~~~~~~~~~~

The first example o

.. code:: python3

    information_topo.information_complex(Ninfomut)


.. image:: images/iris_pairwise_scatter.png

and the multivariate joint-entropies :math:`H_k` just generalise the preceding to k variables:

.. math::	
    H_1=H(X_{j};P)=k\sum_{x \in [N_j] }p(x)\ln p(x) 

