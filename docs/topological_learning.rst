Topological Learning
====================

Topological Learning principles
-------------------------------

Information Complexes
~~~~~~~~~~~~~~~~~~~~~

The presentation of the basic methods and principles we made so far mostly relied on basic information lattice decomposition and simplex structure.
In what follows, we will go one stepp further by introducing to simplicial complexes of information which can display much richer structures. This will be the 
occasion to study more in depth information paths, the analog of homotopical paths in information theory. We will consider subcomplex of this simplicial structure,
invocating the fact that any simplicial complex can be realized as a subcomplex of a simplex (`Steenrod 1947 <https://www.jstor.org/stable/1969172>`_ , p.296).

As introduced previously, An information path :math:`IP_k` of degree k on :math:`I_k` landscape is defined as a sequence of elements of
the lattice that begins at the least element of the lattice (the identity-constant “0”), travels along edges from element to element of 
increasing degree of the lattice and ends at the greatest element of the lattice of degree k (a piecewise linear function). The
first derivative of an :math:`IP_k` path is minus the conditional mutual information. The critical dimension of an :math:`IP_k` path 
is the degree of its first minimum. A positive information path is an information path from 0 to a given :math:`I_k` corresponding to a given 
k-tuple of variables such that :math:`I_k < I_{k-1}  < ... < I_1` (a chain, total order). 
A maximal positive information path is a positive information path of maximal length: it ends at minima of :math:`I_k` along the path (a minima 
of the free energy components quantified by :math:`I_k`. In statistical terms, this minima is equivalent to a conditionnal independence: it means 
that the conditional mutual information (slope) of the paths goes throug 0.  
Those maximal paths identifies the maximal faces of the :math:`I_k` complex and charaterize it, because a simplicial complex is uniquely determined 
by the list of its maximal faces. Hence, the set of all these paths defines uniquely the :math:`I_k` complex (or minimum free energy complex). 
An example of such a complex of dimension 4, with its information path :math:`IP_k`, is given in this figure: 

.. image:: images/info_complex.png

Note that, as a result of classical Shannonian information inequalities, any complex of dimension 3 or below is necessarilly a simplex, indicating 
that in information and statitics, 3 and 4 dimensional topology are also very special. 

Poincaré-Shannon Machine
~~~~~~~~~~~~~~~~~~~~~~~~

Learning - Information sensory processing."Understanding is compressing" Chaitin. Efficient coding (Attneave, 1952): the goal of sensory perception is to extract 
the redundancies and to find the most compressed representation of the environment. Any kind of symmetry and invariance are information redundancies and 
Gestalt principles of perception can be defined on information theoretic terms.

.. image::  images/figure_Attneave-Gestalt_complete.png


Information function as loss function (Boltzmann - Helmoltz machine, maxent Jaynes infomax Linsker nadal parga Sejnowsky ... untill current cross entropy
and focal loss = deformed probability).  

.. image::  images/loss_function.jpg


The models developped here are called the Poincaré-Shannon machine in reverence to `Boltzmann Machine <https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwinjPbMnvjrAhUKzYUKHStSA7gQgAMoAHoECAgQAg&url=http%3A%2F%2Fscholar.google.fr%2Fscholar_url%3Furl%3Dhttps%3A%2F%2Fwww.cs.utoronto.ca%2F~hinton%2Fabsps%2Fcogscibm.pdf%26hl%3Dfr%26sa%3DX%26ei%3Duo5nX8mHM72Ay9YP1OOAiAM%26scisig%3DAAGBfm0MtFqrPZRIBb9G16LNS5kfPdVoFw%26nossl%3D1%26oi%3Dscholarr&usg=AOvVaw29iesHzi-bIRQnf2tYDIH1>`_ , 
`Helmholtz Machine <https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjZ8J7GoPjrAhWhxYUKHZxbB74QFjABegQIBBAB&url=http%3A%2F%2Fwww.gatsby.ucl.ac.uk%2F~dayan%2Fpapers%2Fhm95.pdf&usg=AOvVaw1wOfAfLAIVYS83_2EO-6Fi>`_
and the original `Hopfield's network <https://www.pnas.org/content/79/8/2554>`_, since it implements simplicial homology (see `Poincaré’s Analysis Situs <http://analysis-situs.math.cnrs.fr/-Textes-originaux-.html>`_ , that arguably foundate algebraic topology) 
and information theory in a single framework (see `Shannon’s theory of communication <https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjtrqOXrsPrAhVCrxoKHcBDBrQQgAMoAHoECBIQAg&url=http%3A%2F%2Fscholar.google.fr%2Fscholar_url%3Furl%3Dhttps%3A%2F%2Fpure.mpg.de%2Frest%2Fitems%2Fitem_2383162_7%2Fcomponent%2Ffile_2456978%2Fcontent%26hl%3Dfr%26sa%3DX%26scisig%3DAAGBfm2dgGR4Ly92eRCfhrM1BgCnbIBvBA%26nossl%3D1%26oi%3Dscholarr&usg=AOvVaw0ha99XPEPwgTiv3oMC7PTE>`_ , 
that foundate information theory), applied effectively to empirical data.


Unsupervised topological learning
---------------------------------

Causality challenge dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~

We will illustrate the computation of free energy complex (or :math:`I_k` complex) on the synthetic dataset `LUCAS  (LUng CAncer Simple set) <http://www.causality.inf.ethz.ch/data/LUCAS.html>`_ 
of the  `causality challenge <http://www.causality.inf.ethz.ch/challenge.php>`_. Before trying the code on your computer, you will have to download the file "lucas0_train.csv" 
and to save it on your hard disk (here at the path "/home/pierre/Documents/Data/lucas0_train.csv"), and to put your own path in the following commands with the initialisation
of infotopo's parameters. 

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
        information_topo = infotopo(dimension_max = dimension_max, 
                                dimension_tot = dimension_tot, 
                                sample_size = sample_size, 
                                work_on_transpose = work_on_transpose,
                                nb_of_values = nb_of_values, 
                                sampling_mode = sampling_mode, 
                                deformed_probability_mode = deformed_probability_mode,
                                supervised_mode = supervised_mode, 
                                forward_computation_mode = forward_computation_mode,
                                dim_to_rank = 3, number_of_max_val = 4)


The dataset is composed of 11 variables: 1: Smoking, 2: Yellow_Fingers, 3: Anxiety, 4: Peer_Pressure, 5: Genetics, 6: Attention_Disorder, 7: Born_an_Even_Day,
8: Car_Accident, 9: Fatigue, 10: Allergy, 11: Coughing and the 12th variable of iterest: Lung cancer. 
The (buildin) causality chain relations among those varaibles follow this schema:

.. image:: images/causality_schema_LUCAS0.png


Information Complexes
~~~~~~~~~~~~~~~~~~~~~

To compute (approximation) of the information complex (free-energy complex), you can use the following command:

.. code:: python3

    Ninfomut, Nentropie = information_topo.fit(dataset)
    information_topo.information_complex(Ninfomut)

The method "fit" is just a wrapper of the methods "simplicial_entropies_decomposition" and "simplicial_infomut_decomposition", that is introduced to correspond to
the usual methods of scikit-learn, keras, tensorflow (...). The set of all paths of degree-dimension k is intractable computationally (complexity in :math:`\mathcal{O}(k!)` ). 
In order to bypass this issue, the current method "information_complex" computes a fast local algorithm that selects at each element of degree k of a path, the 
positive information path with maximal or minimal :math:`I_{k+1}` value (equivalently, extremal conditional mutual informations) or stops whenever  
:math:`X_k.I_{k+1} \leq 0` and ranks those paths by their length. No doubt that this approximation is rought and shall be improved (to be done). 
The result on the causality challenge dataset is:

.. image:: images/causality_info_paths.png

and it prints the following paths:

.. parsed-literal::

    The path of maximal mutual-info Nb 1  is : [5, 12, 11, 9, 8, 6, 2, 1, 10, 4], The path of minimal mutual-info Nb 1  is : [7, 2, 11], The path of maximal mutual-info Nb 2  is :[2, 12, 11, 9, 3, 6, 10, 5], The path of minimal mutual-info Nb 2  is : [3, 4, 1], The path of maximal mutual-info Nb 3  is : [1, 2, 12, 11, 9, 3, 6, 10, 5], The path of minimal mutual-info Nb 3  is : [10, 4, 7], The path of maximal mutual-info Nb 4  is : [9, 11, 12, 1, 2, 3, 6, 10, 5], The path of minimal mutual-info Nb 4  is : [4, 3, 1], The path of maximal mutual-info Nb 5  is :[8, 9, 11, 12, 5, 6, 2, 1, 10, 4], The path of minimal mutual-info Nb 5  is : [6, 1, 12] etc..

The first maximal path [5, 12, 11, 9, 8, 6, 2, 1, 10, 4]  as length 10 and the first 5 variables corresponds to the longest causal chain of the data as illustrated bellow. 
The fact that the resulting path is so long is likely due to the generating algorithm used for Lucas, and the last [6,2,1,10,4] errors could be removed by statistical test 
thresholding on conditional mutual information values. The next maximal paths fail to identify the other long causal chain of the data, probably as a consequence of
the rought approximation used by the algorithm. The First two minimal paths [7, 2, 11] and [3, 4, 1] identifies unrelated variables or multiple cause causality scheme.

.. image:: images/causality_info_paths_results.png

.. math::	
    H_1=H(X_{j};P)=k\sum_{x \in [N_j] }p(x)\ln p(x) 

