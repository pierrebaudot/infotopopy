How to Use InfoTopo
===================

Infotopo is a general Machine Learning set of tools gathering Topology 
(Cohomology and Homotopy), statistics and information theory 
(information quantifies statistical structures generically) and 
statistical physics.
It provides a matheamticaly formalised expression of deep network and learning,
and propose anuspervised or supervised learning mode (as a special case of the first).

The raw methods are computationnally consuming due to the intrinsic combinatorial 
nature of the topological tools, even in the simplest case of a simplicial case 
(the general case is based on the much broader partition combinatorics) the 
computational complexity is of the order of 2 at the power n. 
As a consequence, an important part of the tools and methods are dedicated 
to overcome this extensive computation. Among the possible strategies and 
heuristics used or currently developped, are:
_ restrict to simplicial cohomology and combinatorics (done here).
_ possible exploration of only the low dimensional structures (done here).
_ possible exploration of only most or least informative paths (done here).
_ possible restriction 2nd degree-dimension statistical interactions: 
what is computed here is the equivalent of the Cech complex (with all degree-
dimension computed), and such restriction is equivalent to computing the Vietoris-Rips 
complex (in development). 
_ compute on GPU (in development).
As a result, for this 0.1 version of the software, and for computation with 
commercial average PC, we recommand to analyse up to 20 variables (or dimensions)
at a time in the raw brut-force approach (see performance section).



We now present some basic example of use, inspiring our presentation from 
the remarkable presentation of `UMAP by McInnes. <https://umap-learn.readthedocs.io/en/latest/>`_
We first import some few tools: some of the datasets available in sklearn, seaborn to
visualise the results, and pandas to handle the data.

.. code:: python3

    from sklearn.datasets import load_iris, load_digits
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import timeit

.. code:: python3

    sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

Iris data
---------

The first example of dataset application we will present is the `iris
dataset <https://en.wikipedia.org/wiki/Iris_flower_data_set>`__. It is
a very small dataset composed of 4 Random-Variables or dimensions that 
quantify various petals and sepals observables of 3 different species of 
Iris flowers, like petal length, for 150 flowers or points (50 for each 
species). In the context of Infotopo it means that dimension_tot = 4  
and sample_size = 150 (we consider all the points), and as the dimension
of the data set is small we will make the complete analysis of the 
simplicial structure of dependencies by setting the maximum dimension 
of analysis to dimension_max = dimension_tot. We also set the other 
parameters of infotopo to approriate, as further explained.   
We can load the iris dataset from sklearn.

.. code:: python3

    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
    print(iris.DESCR)

    dimension_max = iris.data.shape[1]
    dimension_tot = iris.data.shape[1]
    sample_size = iris.data.shape[0]
    nb_of_values =9
    forward_computation_mode = False
    work_on_transpose = False
    supervised_mode = False
    sampling_mode = 1
    deformed_probability_mode = False
    


.. parsed-literal::

    Iris Plants Database
    ====================
    
    Notes
    -----
    Data Set Characteristics:
        :Number of Instances: 150 (50 in each of three classes)
        :Number of Attributes: 4 numeric, predictive attributes and the class
        :Attribute Information:
            - sepal length in cm
            - sepal width in cm
            - petal length in cm
            - petal width in cm
            - class:
                    - Iris-Setosa
                    - Iris-Versicolour
                    - Iris-Virginica
        :Summary Statistics:
    
        ============== ==== ==== ======= ===== ====================
                        Min  Max   Mean    SD   Class Correlation
        ============== ==== ==== ======= ===== ====================
        sepal length:   4.3  7.9   5.84   0.83    0.7826
        sepal width:    2.0  4.4   3.05   0.43   -0.4194
        petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
        petal width:    0.1  2.5   1.20  0.76     0.9565  (high!)
        ============== ==== ==== ======= ===== ====================
    
        :Missing Attribute Values: None
        :Class Distribution: 33.3% for each of 3 classes.
        :Creator: R.A. Fisher
        :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
        :Date: July, 1988
    
    This is a copy of UCI ML iris datasets.
    http://archive.ics.uci.edu/ml/datasets/Iris
    
    The famous Iris database, first used by Sir R.A Fisher
    
    This is perhaps the best known database to be found in the
    pattern recognition literature.  Fisher's paper is a classic in the field and
    is referenced frequently to this day.  (See Duda & Hart, for example.)  The
    data set contains 3 classes of 50 instances each, where each class refers to a
    type of iris plant.  One class is linearly separable from the other 2; the
    latter are NOT linearly separable from each other.
    
    References
    ----------
       - Fisher,R.A. "The use of multiple measurements in taxonomic problems"
         Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to
         Mathematical Statistics" (John Wiley, NY, 1950).
       - Duda,R.O., & Hart,P.E. (1973) Pattern Classification and Scene Analysis.
         (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.
       - Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System
         Structure and Classification Rule for Recognition in Partially Exposed
         Environments".  IEEE Transactions on Pattern Analysis and Machine
         Intelligence, Vol. PAMI-2, No. 1, 67-71.
       - Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE Transactions
         on Information Theory, May 1972, 431-433.
       - See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al"s AUTOCLASS II
         conceptual clustering system finds 3 classes in the data.
       - Many, many more ...
    

As visualizing data in 4 dimensions or more is hard or not possible, we can first 
plot all the pairwise scatterplot matrix to present the pairwise correlations and 
dependencies between the variables, using Seaborn and pandas dataframe.

.. code:: python3

    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['species'] = pd.Series(iris.target).map(dict(zip(range(3),iris.target_names)))
    sns.pairplot(iris_df, hue='species')
    plt.show()


.. image:: images/iris_pairwise_scatter.png


All those 2D views gives a rought but misleading idea of what the data looks 
like in high dimension since, as we will see, some fully emergent  
statistical dependences (synergic) can appear in higher dimension which are 
totally unobservable in those 2D views. However such 2D views gives a fair
visual estimation of how much each pairs of variale covary, the correlation 
coefficient and its generalization to non-linear relations, the pairwise 
Mutual Information (I2). In topological Data Analysis terms, it gives rought 
idea of what the skeleton of a Vietoris-Rips (information or correlation) complex
of the data could be.
We will see how to go beyond this pairwise statistical interaction case, and how
we can unravel some purely emergent higher dimensional interations. Along this 
way, we will see how to compute and estimate all classical information functions,
multivariate Entropies, Mutual Informations and Conditional Entropies and 
Mutual Informations. 

To use infotopo we need to first construct a infotopo object from 
the infotopo package. This makes a lot of same word, information is a 
functor, a kind of general application or map, that could be either a 
function or a class. So let's first import the infotopo library, we a set 
of specifications of the parametters (cf. section parameters, some of them 
like dimension_max = dimension_tot and sample_size have been fixed 
previously to the size of the data input matrix).

.. code:: python3

    import infotopo

.. code:: python3

    information_topo = infotopo.infotopo(dimension_max = dimension_max, 
                                dimension_tot = dimension_tot, 
                                sample_size = sample_size, 
                                work_on_transpose = work_on_transpose,
                                nb_of_values = nb_of_values, 
                                sampling_mode = sampling_mode, 
                                deformed_probability_mode = deformed_probability_mode,
                                supervised_mode = supervised_mode, 
                                forward_computation_mode = forward_computation_mode)

Now we will compute all the simplicial semi-lattice of marginal and joint-entropy, 
that contains 2 power n elements including the unit 0 reference measure element
The figure below give the usual Venn diagrams representation of set theoretic unions 
and the corresponding semi-lattice of joint Random Variables and Joint Entropies, together 
with its correponding simplicial representation, for 3 (top) and 4 variables-dimension 
(bottom, the case of the iris dataset with 2 power 4 joint random variables). The edges of
the lattice are in one to one correspondence with conditional entropies.   

.. image:: images/figure_lattice.png

To do this we will call simplicial_entropies_decomposition, that gives in output 
all the joint entropies in the form of a dictionary with keys given by the tuple of 
the joint variables (ex: (1,3,4)) and  with values the joint or marginal entropy in bit 
(presented below).

.. code:: python3

    Nentropie = information_topo.simplicial_entropies_decomposition(iris.data)


.. parsed-literal::

    {(4,): 2.9528016441309237, (3,): 2.4902608474907497, (2,): 2.5591245822618114, (1,): 2.8298425472847066, (3, 4): 3.983309507504916, (2, 4): 4.798319817958397, (1, 4): 4.83234271597051, (2, 3): 4.437604597473526, (1, 3): 4.2246575340121835, (1, 2): 4.921846615158947, (2, 3, 4): 5.561696151051504, (1, 3, 4): 5.426426190681815, (1, 2, 4): 6.063697650692486, (1, 2, 3): 5.672729631265195, (1, 2, 3, 4): 6.372515544003377}



Such dictionary is hard to read; to allow a relevant visualization of the
the simplicial entropy structure, the function simplicial_entropies_decomposition
also plots the Entropy landscapes  
``embedding`` as a standard scatterplot and color by the target array
(since it applies to the transformed data which is in the same order as
the original).



This concludes our introduction to basic infotopo usage -- hopefully this
has given you the tools to get started for yourself. Further tutorials,
covering infotopo parameters and more advanced usage are also available when
you wish to dive deeper.