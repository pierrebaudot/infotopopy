.. infotopo documentation master file, created by
   sphinx-quickstart on Tue May 26 00:08:49 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

InfoTopo: Topological Information Data Analysis. Deep statistical unsupervised and supervised learning.

InfoTopo is a Machine Learning method based on Information Cohomology, a cohomology of statistical systems [1,6,7]. 
It allows to estimate higher order statistical structures, dependences and (refined) independences or generalised (possibly non-linear) correlations 
and to uncover their structure as simplicial complex.
It provides estimations of the basic information functions, entropy, joint and condtional, multivariate Mutual-Informations and conditional MI, Total Correlations...
InfoTopo is at the cross-road of Topological Data Analysis, Deep Neural Network learning and statistical physics:
1. With respect to Topological Data Analysis (TDA), it provides intrinsically probabilistic methods that does not assume metric (Random Variable's alphabets are not necessarilly ordinal).
2. With respect to Deep Neural Networks (DNN), it provides a simplical complex constrained DNN structure with topologically derived unsupervised and supervised 
learning rules (forward propagation, differential statistical operators) 
3. With respect to statistical physics, it provides generalized correlation functions, free and internal energy functions, estimations of the n-body interactions 
contributions to energy functional, that holds in non-homogeous and finite-discrete case, without mean-field assumptions. Cohomological Complex implements the minimum free-energy principle.
Information Topology is rooted in cognitive sciences and computational neurosciences, and generalizes-unifies some consciousness theories [5].

It assumes basically:
1. a classical probability space (here a discrete finite sample space), geometrically formalized as a probability simplex with basic conditionning and Bayes rule and implementing  
2. a complex (here simplicial) of random variable with a joint operators
3. a quite generic coboundary operator (Hochschild, Homological algebra with a (left) action of conditional expectation)

The details for the underlying mathematics and methods can be found in the papers:
[1] Vigneaux J., Topology of Statistical Systems. A Cohomological Approach to Information Theory. Ph.D. Thesis, Paris 7 Diderot University, Paris, France, June 2019. `PDF <https://webusers.imj-prg.fr/~juan-pablo.vigneaux/these.pdf>`
[2] Baudot P., Tapia M., Bennequin, D. , Goaillard J.M., Topological Information Data Analysis. 2019, Entropy, 21(9), 869  `PDF <https://www.mdpi.com/1099-4300/21/9/869>`
[3] Baudot P., The Poincaré-Shannon Machine: Statistical Physics and Machine Learning aspects of Information Cohomology. 2019, Entropy , 21(9),  `PDF <https://www.mdpi.com/1099-4300/21/9/881>`
[4] Tapia M., Baudot P., Dufour M., Formizano-Treziny C., Temporal S., Lasserre M., Kobayashi K., Goaillard J.M.. Neurotransmitter identity and electrophysiological phenotype are genetically coupled in midbrain dopaminergic neurons. Scientific Reports. 2018. `PDF <https://www.nature.com/articles/s41598-018-31765-z>`
[5] Baudot P., Elements of qualitative cognition: an Information Topology Perspective. Physics of Life Reviews. 2019. extended version on Arxiv. `PDF <https://arxiv.org/abs/1807.04520>`
[6] Baudot P., Bennequin D., The homological nature of entropy. Entropy, 2015, 17, 1-66; doi:10.3390. `PDF <https://www.mdpi.com/1099-4300/17/5/3253>`
[7] Baudot P., Bennequin D., Topological forms of information. AIP conf. Proc., 2015. 1641, 213. `PDF <https://aip.scitation.org/doi/abs/10.1063/1.4905981>`

====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
https://www.mdpi.com/1099-4300/17/5/3253