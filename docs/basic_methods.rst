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

To use UMAP for this task we need to first construct a UMAP object that
will do the job for us. That is as simple as instantiating the class. So
let's import the umap library and do that.

.. code:: python3

    import infotopo

.. code:: python3

    reducer = umap.UMAP()

Now we need to train our reducer, letting it learn about the manifold.
For this UMAP follows the sklearn API and has a method ``fit`` which we
pass the data we want the model to learn from. Since, at the end of the
day, we are going to want to reduced representation of the data we will
use, instead, the ``fit_transform`` method which first calls ``fit`` and
then returns the transformed data as a numpy array.

.. code:: python3

    embedding = reducer.fit_transform(iris.data)
    embedding.shape


.. parsed-literal::

    (150, 2)



The result is an array with 150 samples, but only two feature columns
(instead of the four we started with). This is because, by default, UMAP
reduces down to 2D. Each row of the array is a 2-dimensional
representation of the corresponding flower. Thus we can plot the
``embedding`` as a standard scatterplot and color by the target array
(since it applies to the transformed data which is in the same order as
the original).

.. code:: python3

    plt.scatter(embedding[:, 0], embedding[:, 1], c=[sns.color_palette()[x] for x in iris.target])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the Iris dataset', fontsize=24);




.. image:: images/BasicUsage_13_1.png


This does a useful job of capturing the structure of the data, and as
can be seen from the matrix of scatterplots this is relatively accurate.
Of course we learned at least this much just from that matrix of
scatterplots -- which we could do since we only had four different
dimensions to analyse. If we had data with a larger number of dimensions
the scatterplot matrix would quickly become unwieldy to plot, and far
harder to interpret. So moving on from the Iris dataset, let's consider
the digits dataset.

Digits data
-----------

First we will load the dataset from sklearn.

.. code:: python3

    digits = load_digits()
    print(digits.DESCR)


.. parsed-literal::

    Optical Recognition of Handwritten Digits Data Set
    ===================================================
    
    Notes
    -----
    Data Set Characteristics:
        :Number of Instances: 5620
        :Number of Attributes: 64
        :Attribute Information: 8x8 image of integer pixels in the range 0..16.
        :Missing Attribute Values: None
        :Creator: E. Alpaydin (alpaydin '@' boun.edu.tr)
        :Date: July; 1998
    
    This is a copy of the test set of the UCI ML hand-written digits datasets
    http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits
    
    The data set contains images of hand-written digits: 10 classes where
    each class refers to a digit.
    
    Preprocessing programs made available by NIST were used to extract
    normalized bitmaps of handwritten digits from a preprinted form. From a
    total of 43 people, 30 contributed to the training set and different 13
    to the test set. 32x32 bitmaps are divided into nonoverlapping blocks of
    4x4 and the number of on pixels are counted in each block. This generates
    an input matrix of 8x8 where each element is an integer in the range
    0..16. This reduces dimensionality and gives invariance to small
    distortions.
    
    For info on NIST preprocessing routines, see M. D. Garris, J. L. Blue, G.
    T. Candela, D. L. Dimmick, J. Geist, P. J. Grother, S. A. Janet, and C.
    L. Wilson, NIST Form-Based Handprint Recognition System, NISTIR 5469,
    1994.
    
    References
    ----------
      - C. Kaynak (1995) Methods of Combining Multiple Classifiers and Their
        Applications to Handwritten Digit Recognition, MSc Thesis, Institute of
        Graduate Studies in Science and Engineering, Bogazici University.
      - E. Alpaydin, C. Kaynak (1998) Cascading Classifiers, Kybernetika.
      - Ken Tang and Ponnuthurai N. Suganthan and Xi Yao and A. Kai Qin.
        Linear dimensionalityreduction using relevance weighted LDA. School of
        Electrical and Electronic Engineering Nanyang Technological University.
        2005.
      - Claudio Gentile. A New Approximate Maximal Margin Classification
        Algorithm. NIPS. 2000.
    


We can plot a number of the images to get an idea of what we are looking
at. This just involves matplotlib building a grid of axes and then
looping through them plotting an image into each one in turn.

.. code:: python3

    fig, ax_array = plt.subplots(20, 20)
    axes = ax_array.flatten()
    for i, ax in enumerate(axes):
        ax.imshow(digits.images[i], cmap='gray_r')
    plt.setp(axes, xticks=[], yticks=[], frame_on=False)
    plt.tight_layout(h_pad=0.5, w_pad=0.01)



.. image:: images/BasicUsage_18_0.png


As you can see these are quite low resolution images -- for the most
part they are recognisable as digits, but there are a number of cases
that are sufficiently blurred as to be questionable even for a human to
guess at. The zeros do stand out as the easiest to pick out as notably
different and clearly zeros. Beyond that things get a little harder:
some of the squashed thing eights look awfully like ones, some of the
threes start to look a little like crossed sevens when drawn badly, and
so on.

Each image can be unfolded into a 64 element long vector of grayscale
values. It is these 64 dimensional vectors that we wish to analyse: how
much of the digits structure can we discern? At least in principle 64
dimensions is overkill for this task, and we would reasonably expect
that there should be some smaller number of "latent" features that would
be sufficient to describe the data reasonably well. We can try a
scatterplot matrix -- in this case just of the first 10 dimensions so
that it is at least plottable, but as you can quickly see that approach
is not going to be sufficient for this data.

.. code:: python3

    digits_df = pd.DataFrame(digits.data[:,:10])
    digits_df['digit'] = pd.Series(digits.target).map(lambda x: 'Digit {}'.format(x))
    sns.pairplot(digits_df, hue='digit', palette='Spectral');



.. image:: images/BasicUsage_20_1.png


In contrast we can try using UMAP again. It works exactly as before:
construct a model, train the model, and then look at the transformed
data. TO demonstrate more of UMAP we'll go about it differently this
time and simply use the ``fit`` method rather than the ``fit_transform``
approach we used for Iris.

.. code:: python3

    reducer = umap.UMAP(random_state=42)
    reducer.fit(digits.data)


.. parsed-literal::

    UMAP(a=1.576943460405378, alpha=1.0, angular_rp_forest=False,
       b=0.8950608781227859, bandwidth=1.0, gamma=1.0, init='spectral',
       local_connectivity=1.0, metric='euclidean', metric_kwds={},
       min_dist=0.1, n_components=2, n_epochs=None, n_neighbors=15,
       negative_sample_rate=5, random_state=42, set_op_mix_ratio=1.0,
       spread=1.0, target_metric='categorical', target_metric_kwds={},
       transform_queue_size=4.0, transform_seed=42, verbose=False)



Now, instead of returning an embedding we simply get back the reducer
object, now having trained on the dataset we passed it. To access the
resulting transform we can either look at the ``embedding_`` attribute
of the reducer object, or call transform on the original data.

.. code:: python3

    embedding = reducer.transform(digits.data)
    # Verify that the result of calling transform is 
    # idenitical to accessing the embedding_ attribute
    assert(np.all(embedding == reducer.embedding_))
    embedding.shape




.. parsed-literal::

    (1797, 2)



We now have a dataset with 1797 rows (one for each hand-written digit
sample), but only 2 columns. As with the Iris example we can now plot
the resulting embedding, coloring the data points by the class that
they belong to (i.e. the digit they represent).

.. code:: python3

    plt.scatter(embedding[:, 0], embedding[:, 1], c=digits.target, cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
    plt.title('UMAP projection of the Digits dataset', fontsize=24);




.. image:: images/BasicUsage_26_1.png


We see that UMAP has successfully captured the digit classes. There are
also some interesting effects as some digit classes blend into one
another (see the eights, ones, and sevens, with some nines in between),
and also cases where digits are pushed away as clearly distinct (the
zeros on the right, the fours at the top, and a small subcluster of ones
at the bottom come to mind). To get a better idea of why UMAP chose to
do this it is helpful to see the actual digits involve. One can do this
using `bokeh <https://bokeh.pydata.org/en/latest/>`__ and mouseover
tooltips of the images.

First we'll need to encode all the images for inclusion in a dataframe.

.. code:: python3

    from io import BytesIO
    from PIL import Image
    import base64



.. code:: python3

    def embeddable_image(data):
        img_data = 255 - 15 * data.astype(np.uint8)
        image = Image.fromarray(img_data, mode='L').resize((64, 64), Image.BICUBIC)
        buffer = BytesIO()
        image.save(buffer, format='png')
        for_encoding = buffer.getvalue()
        return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()



Next we need to load up bokeh and the various tools from it that will be
needed to generate a suitable interactive plot.

.. code:: python3

    from bokeh.plotting import figure, show, output_notebook
    from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
    from bokeh.palettes import Spectral10
    
    output_notebook()



.. raw:: html

    
        <div class="bk-root">
            <a href="https://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
            <span id="0a99b177-0c09-4f23-9312-177e9a03b8be">Loading BokehJS ...</span>
        </div>




Finally we generate the plot itself with a custom hover tooltip that
embeds the image of the digit in question in it, along with the digit
class that the digit is actually from (this can be useful for digits
that are hard even for humans to classify correctly).

.. code:: python3

    digits_df = pd.DataFrame(embedding, columns=('x', 'y'))
    digits_df['digit'] = [str(x) for x in digits.target]
    digits_df['image'] = list(map(embeddable_image, digits.images))
    
    datasource = ColumnDataSource(digits_df)
    color_mapping = CategoricalColorMapper(factors=[str(9 - x) for x in digits.target_names],
                                           palette=Spectral10)
    
    plot_figure = figure(
        title='UMAP projection of the Digits dataset',
        plot_width=600,
        plot_height=600,
        tools=('pan, wheel_zoom, reset')
    )
    
    plot_figure.add_tools(HoverTool(tooltips="""
    <div>
        <div>
            <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
        </div>
        <div>
            <span style='font-size: 16px; color: #224499'>Digit:</span>
            <span style='font-size: 18px'>@digit</span>
        </div>
    </div>
    """))
    
    plot_figure.circle(
        'x',
        'y',
        source=datasource,
        color=dict(field='digit', transform=color_mapping),
        line_alpha=0.6,
        fill_alpha=0.6,
        size=4
    )
    show(plot_figure)


.. raw:: html
   :file: basic_usage_bokeh_example.html

As can be seen, the nines that blend between the ones and the sevens are
odd looking nines (that aren't very rounded) and do, indeed, interpolate
surprisingly well between ones with hats and crossed sevens. In contrast
the small disjoint cluster of ones at the bottom of the plot is made up
of ones with feet (a horizontal line at the base of the one) which are,
indeed, quite distinct from the general mass of ones.

This concludes our introduction to basic UMAP usage -- hopefully this
has given you the tools to get started for yourself. Further tutorials,
covering UMAP parameters and more advanced usage are also available when
you wish to dive deeper.