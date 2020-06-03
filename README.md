INFOTOPO

Programs for Information Topology Data Analysis INFOTOPO : new rewrited version of the 2017 scripts available at https://github.com/pierrebaudot/INFOTOPO/
 
Programs for Information Topology Data Analysis Information Topology is a program written in Python (compatible with Python 3.4.x), with a graphic interface built using TKinter [1], plots drawn using Matplotlib [2], calculations made using NumPy [3], and scaffold representations drawn using NetworkX [4]. It computes all the results on information presented in the study [5,6,7,8,9], that is all the usual information functions: entropy, joint entropy between k random variables (Hk), mutual informations between k random variables (Ik), conditional entropies and mutual informations and provides their cohomological (and homotopy) visualisation in the form of information landscapes and information paths together with an approximation of the minimum information energy complex [5,6,7,8,9]. It is applicable on any set of empirical data that is data with several trials-repetitions-essays (parameter m), and also allows to compute the undersampling regime, the degree k above which the sample size m is to small to provide good estimations of the information functions [5,6,7,8,9]. The computational exploration is restricted to the simplicial sublattice of random variable (all the subsets of k=n random variables) and has hence a complexity in O(2^n). In this simplicial setting we can exhaustively estimate information functions on the simplicial information structure, that is joint-entropy Hk and mutual-informations Ik at all degrees k=<n and for every k-tuple, with a standard commercial personal computer (a laptop with processor Intel Core i7-4910MQ CPU @ 2.90GHz * 8) up to k=n=21 in reasonable time (about 3 hours). Using the expression of joint-entropy and the probability obtained using equation and marginalization [5], it is possible to compute the joint-entropy and marginal entropy of all the variables. The alternated expression of n-mutual information given by equation then allows a direct evaluation of all of these quantities. The definitions, formulas and theorems are sufficient to obtain the algorithm [5]. We will further develop a refined interface but for the moment it works like this, and requires minimum Python use knowledge. Please contact pierre.baudot [at] gmail.com for questions, request, developments (etc.). The version V1.2, includes more commentaries and notes in the programs, the previous readfile has been included in the visualization, energy vs entropy landscapes have been added to vizualisation, the shuffles for statistical dependence test has been implemented in COMPUTATION and VISUALIZATION, The determination of the undersampling dimension :



[1] J.W. Shipman. Tkinter reference: a gui for python. . New Mexico Tech Computer Center, Socorro, New Mexico, 2010. 
[2] J.D. Hunter. Matplotlib: a 2d graphics environment. Comput. Sci. Eng., 9:22–30, 2007. 
[3] S. Van Der Walt, C. Colbert, and G. Varoquaux. The numpy array: a structure for efficient numerical computation. Comput. Sci. Eng., 13:22– 30, 2011. [4] A.A. Hagberg, D.A. Schult, and P.J. Swart. Exploring network structure, dynamics, and function using networkx. Proceedings of the 7th Python in Science Conference (SciPy2008). Gel Varoquaux, Travis Vaught, and Jarrod Millman (Eds), (Pasadena, CA USA), pages 11–15, 2008. 
[5] Baudot P., Tapia M., Bennequin, D. , Goaillard J.M., Topological Information Data Analysis. 2019, Entropy, 21(9), 869    PDF
[6] Baudot P., The Poincaré-Shannon Machine: Statistical Physics and Machine Learning aspects of Information Cohomology. 2019, Entropy , 21(9),  PDF
[7] Tapia M., Baudot P., Dufour M., Formizano-Treziny C., Temporal S., Lasserre M., Kobayashi K., Goaillard J.M.. Neurotransmitter identity and electrophysiological phenotype are genetically coupled in midbrain dopaminergic neurons. Scientific Reports. 2018. PDF. BioArXiv168740. PDF
[8] Baudot P., Elements of qualitative cognition: an Information Topology Perspective. Physics of Life Reviews. 2019 LINK. extended version on Arxiv. PDF
[9] Baudot P., Bennequin D., The homological nature of entropy. Entropy, 2015, 17, 1-66; doi:10.3390. PDF
[10] Baudot P., Bennequin D., Topological forms of information. AIP conf. Proc., 2015. 1641, 213. PDF









Previous Package available at https://github.com/pierrebaudot/INFOTOPO 

Programs for Information Topology Data Analysis
INFOTOP_V1.2

Programs for Information Topology Data Analysis Information Topology is a program written in Python (compatible with Python 3.4.x), with a graphic interface built using TKinter [1], plots drawn using Matplotlib [2], calculations made using NumPy [3], and scaffold representations drawn using NetworkX [4]. It computes all the results on information presented in the study [5], that is all the usual information functions: entropy, joint entropy between k random variables (Hk), mutual informations between k random variables (Ik), conditional entropies and mutual informations and provides their cohomological (and homotopy) visualisation in the form of information landscapes and information paths together with an approximation of the minimum information energy complex [5]. It is applicable on any set of empirical data that is data with several trials-repetitions-essays (parameter m), and also allows to compute the undersampling regime, the degree k above which the sample size m is to small to provide good estimations of the information functions [5]. The computational exploration is restricted to the simplicial sublattice of random variable (all the subsets of k=n random variables) and has hence a complexity in O(2^n). In this simplicial setting we can exhaustively estimate information functions on the simplicial information structure, that is joint-entropy Hk and mutual-informations Ik at all degrees k=<n and for every k-tuple, with a standard commercial personal computer (a laptop with processor Intel Core i7-4910MQ CPU @ 2.90GHz * 8) up to k=n=21 in reasonable time (about 3 hours). Using the expression of joint-entropy and the probability obtained using equation and marginalization [5], it is possible to compute the joint-entropy and marginal entropy of all the variables. The alternated expression of n-mutual information given by equation then allows a direct evaluation of all of these quantities. The definitions, formulas and theorems are sufficient to obtain the algorithm [5]. We will further develop a refined interface but for the moment it works like this, and requires minimum Python use knowledge. Please contact pierre.baudot [at] gmail.com for questions, request, developments (etc.). The version V1.2, includes more commentaries and notes in the programs, the previous readfile has been included in the visualization, energy vs entropy landscapes have been added to vizualisation, the shuffles for statistical dependence test has been implemented in COMPUTATION and VISUALIZATION, The determination of the undersampling dimension :

INFOTOPO is currently divided into 3 programs:

    INFOTOPO_COMPUTATION_V1.2.py 
This program computes all the information quantities for all k-tuples below n=Nb_var and save them in object file. The input is an excel (.xlsx) table containing the data values, e.g. the matrix D with first row and column containing the labels, the rows are the random variables (computation with usual PC up to n=21 rows , n=Nb_var=Nb_vartot) and the columns are the differents trials-repetitions-essays (parameter m). It first estimate the joint probability density at a given grianing-resampling of the variables (parameter N=Nb_bins) [5]. It prints the overall histograms of raw and resampled values and of the raw and resampled matrix. The information functions are then estimated for each k-tuples and saved in object-files: _ 'ENTROPY'.pkl save the object Nentropy: a dictionaries (x,y) with x a list of kind (1,2,5) and y a Hk value in bit. It contains the 2^n values of joint entropies _ 'ENTROPY_ORDERED'.pkl save the object Nentropy_per_order_ordered, the ordered dictionary of Nentropy where the order is given by the entropy values. _ 'INFOMUT'.plk save the object Ninfomut: a dictionaries (x,y) with x a list of kind (1,2,5) and y a Ik value in bit; It contains the 2^n values of mutual informations. _ 'ENTROPY_SUM'.pkl save the object entropy_sum_order: a dictionaries (k,y) with k the degree and y the mean Hk value over all k-tuple in bit _ 'INFOMUT_ORDERED'.pkl save the object Ninfomut_per_order_ordered, the odered dictionary of Ninfomut where the order is given by the infomut values. _ 'INFOMUT_ORDEREDList'.plk save the object infomut_per_order: the same as INFOMUT_ORDERED but saved as a list. _ 'INFOMUT_SUM'.pkl save the object Infomut_sum_order: a dictionaries (k,y) with k the degree and y the mean Ik value over all k-tuple in bit.
For statistical test of dependence (compute_shuffle), it generates a number (nb_of_shuffle=XXX) shuffles that preserves the marginal distributions and make the computation of all the information structure for them giving in output all the files with 'ENTROPYXXX'.pkl, 'ENTROPY_ORDEREDXXX'.pkl, 'INFOMUTXXX'.plk (...) with the number XXX of the shuffle at the end of the extension. 
It also allows to compute all the information structure  for Nb_of_N=XXXX values of the Graining N (compute_different_N) in output all the files with 'ENTROPYXXX'.pkl, 'ENTROPY_ORDEREDXXX'.pkl, 'INFOMUTXXX'.plk (...) with the number XXX of the shuffle at the end of the extension. 
It also allows to compute all the information structure  for Nb_of_m=XXXX values of the sample size m (compute_different_m) in output all the files with 'ENTROPYXXX'.pkl, 'ENTROPY_ORDEREDXXX'.pkl, 'INFOMUTXXX'.plk (...) with the number XXX of the shuffle at the end of the extension. 


    INFOTOPO_VISUALIZATION_V1.1.py 
This program computes all the visualization of information quantities in the form of distributions, information landscapes, mean landscapes and information paths together with an approximation of the minimum information energy complex, and scafolds of the Information. The input is the saved object-files .plk. It also allows to visualize the result of the test, the parameters are the same as previously exposed. You have to choose the plk to load at the begining of the program and the corresponding figures of output you want by assigning a boolean value:
Computes and display various entropies, means,  efficiencies etc...    
SHOW_results_ENTROPY 

Computes and display various infomut, means, etc...    
SHOW_results_INFOMUT 

Computes and display cond infomut, cond infomut landscape, etc... 
SHOW_results_COND_INFOMUT 

Computes and display HISTOGRAMS ENTROPY, ENTROPY landscape, etc...
SHOW_results_ENTROPY_HISTO 

Computes and display INFOMUT HISTOGRAMS , INFOMUT landscape, etc...
SHOW_results_INFOMUT_HISTO 

Computes and display INFOMUT PATHS, etc...
SHOW_results_INFOMUT_path

Computes and display SCAFFOLDS (RING representation) of mutual info
currently only for I2 ( pairwise infomut...), Ik to be developped soon
SHOW_results_SCAFOLD 

Computes and display INFOMUT-ENTROPY-k landscape (ENERGY VS. ENTROPY) ...
SHOW_results_INFO_entropy_landscape

Computes the mean Ik as a function os the binning graining N and dim k ..
SHOW_results_INFOMUT_Normed_per_bins

Computes the mean Ik as a function os the sample size m and dim k ..
SHOW_results_INFOMUT_Normed_per_samplesize 

Reload the information landscapes saved in pkl file
SHOW_results_RELOAD_LANDSCAPES 

Print data saved in pkl file
SHOW_results_PRINT_PLK_file  



[1] J.W. Shipman. Tkinter reference: a gui for python. . New Mexico Tech Computer Center, Socorro, New Mexico, 2010. [2] J.D. Hunter. Matplotlib: a 2d graphics environment. Comput. Sci. Eng., 9:22–30, 2007. [3] S. Van Der Walt, C. Colbert, and G. Varoquaux. The numpy array: a structure for efficient numerical computation. Comput. Sci. Eng., 13:22– 30, 2011. [4] A.A. Hagberg, D.A. Schult, and P.J. Swart. Exploring network structure, dynamics, and function using networkx. Proceedings of the 7th Python in Science Conference (SciPy2008). Gel Varoquaux, Travis Vaught, and Jarrod Millman (Eds), (Pasadena, CA USA), pages 11–15, 2008. [5] M. Tapia, P. Baudot, M. Dufour, C. Formisano-Tréziny, S. Temporal, M. Lasserre, J. Gabert, K. Kobayashi, JM. Goaillard . Information topology of gene expression profile in dopaminergic neurons doi: https://doi.org/10.1101/168740 http://www.biorxiv.org/content/early/2017/07/26/168740