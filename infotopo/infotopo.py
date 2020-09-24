#!/usr/bin/env python
# coding: utf-8


# list of dependencies
#import
import math
import numpy as np
import itertools
from itertools import combinations, chain
import logging
from decimal import Decimal
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import networkx as nx
from collections import OrderedDict
import heapq
from operator import itemgetter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


###################################################################################
################             COMPUTE INFOPATH               #######################
###################################################################################

def compute_info_path(data_mat, dimension_max, dimension_tot, nbtrials):
    Nentropie={}
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger("compute info_path")
    print("Percent of tuples processed : 0")
    # Compute all pairs of entropy and mutual informations
    dimension_max_temp = dimension_max + 0
    dimension_max = 2
    allsubsets = lambda n: list(chain(*[combinations(range(1,n), ni) for ni in range(dimension_max+1)]))
    list_tuples=allsubsets(dimension_tot+1)
    del list_tuples[0]
    if dimension_max != dimension_tot :
         tot_numb=0
         for  xxx in range(1,dimension_max+1):
                tot_numb=tot_numb + self._binomial(dimension_tot,xxx)
    counter=0
    for tuple_var in list_tuples:
        counter=counter+1
        if dimension_max == dimension_tot:
             if counter % int(pow(2, dimension_max) / 100) == 0:
                 logger.info("PROGRESS: at percent #%i"  % (100*counter/pow(2,dimension_max)))
        else:
             if counter % int(tot_numb / 100) == 0:
                 logger.info("PROGRESS: at percent #%i"  % (100*counter/tot_numb))
        for x in range(0,len(tuple_var)):
           if x==0:
               matrix_temp=np.reshape(data_mat[:,tuple_var[x]-1],(data_mat[:,tuple_var[x]-1].shape[0],1))
           else:
               matrix_temp=np.concatenate((matrix_temp,np.reshape(data_mat[:,tuple_var[x]-1],(data_mat[:,tuple_var[x]-1].shape[0],1))),axis=1)
        probability = self._compute_probability(matrix_temp)
        for x,y in probability.items():
                Nentropie[tuple_var]=Nentropie.get(tuple_var,0) + self._information(probability[x])
    Ninfomut={}
    for x,y in Nentropie.items():
        for k in range(1, len(x)+1):
            for subset in itertools.combinations(x, k):
                Ninfomut[x]=Ninfomut.get(x,0)+ ((-1)**(len(subset)+1))*Nentropie[subset]

     # find the pair of maximum information
    max_info_temp = 0
    for x,y in Ninfomut.items():
        if len(x) == 2 :
            if y > max_info_temp:
                max_info_temp = y
                tuple_maxinfo = x
    print("The pair with Maximum mutual info is :", tuple_maxinfo," with info: ", max_info_temp)

    # Compute all k-uplets of maximum information
    max_info_next = 0
    current_dim = 2
    while  max_info_temp >   max_info_next :
        max_info_temp = max_info_next
        print("DImension of the analysis ", current_dim+1)
        for xxx in range(1,dimension_tot+1):
            if xxx not in tuple_maxinfo:
                tuple_maxinfo=tuple_maxinfo + (xxx,)
                print("tupple in computation ", tuple_maxinfo)
                for k in range(1, len(tuple_maxinfo)+1):
                    print("Number of variable added ", k)
                    for subset in itertools.combinations(tuple_maxinfo, k):
                        print("subset ", subset)
                        if subset not in Nentropie:
                            print("subset not in Nentropie ", subset)
                            for x in range(0,len(subset)):
                                if x==0:
                                    matrix_temp=np.reshape(data_mat[:,subset[x]-1],(data_mat[:,subset[x]-1].shape[0],1))
                                else:
                                    matrix_temp=np.concatenate((matrix_temp,np.reshape(data_mat[:,subset[x]-1],(data_mat[:,subset[x]-1].shape[0],1))),axis=1)
                            probability = self._compute_probability(matrix_temp)
                            for x,y in probability.items():
                                Nentropie[subset]=Nentropie.get(subset,0) + self._information(probability[x])
                        Ninfomut[tuple_maxinfo]=Ninfomut.get(tuple_maxinfo,0)+ ((-1)**(len(subset)+1))*Nentropie[subset]
                        print(" Ninfomut of ",tuple_maxinfo," is ", Ninfomut[tuple_maxinfo])
        max_info_next =-100000
        for a,b in Ninfomut.items():
            if len(a) == current_dim :
                if b > max_info_next:
                    max_info_next = b
                    tuple_maxinfo = a
        print("The ", current_dim,"-tuple with Maximum mutual info is :", tuple_maxinfo," with info: ",max_info_next)
        current_dim = current_dim + 1

    return  Nentropie, Ninfomut




###################################################################################
################             CLASS INFOTOPO                 #######################
###################################################################################

class Infotopo:
    """Compute the simplicial information cohomology of a set of variable.

    This class can be used to compute the simplicial information cohomology of
    a set of variablenotably the joint and conditional entropies, the mutual
    and conditional mutual information, total correlations, and information
    paths within the simplicial set.

    Parameters
    ----------
    dimension_max : int | 16
        Maximum number of random variable (column or dimension) for the
        exploration of the cohomology and lattice
    dimension_tot : int | 16
        Total number of random variable (column or dimension) to consider in
        the input matrix for analysis (the first columns)
    sample_size : int | 1000
        Total number of points (rows or number of trials)  to consider in the
        input matrix for analysis (the first rows)
    work_on_transpose : bool | False
        If True take the transpose of the input matrix (this change column into
        rows etc.)
    nb_of_values : int | 9
        Number of different values for the sampling of each variable (alphabet
        size)
    sampling_mode : int | None
        Define the sampling mode. Use either :

            * 1 : normalization taking the max and min of each columns
              (normalization row by columns)
            * 2 : normalization taking the max and min of the whole matrix
            * 3 : TO BE DEFINED <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    deformed_probability_mode : bool | False
        Mode for the deformed probabilities. Choose either :

            * True : it will compute the "escort distribution" also called the
              "deformed probabilities". p(n,k)= p(k)^n/ (sum(i)p(i)^n, where n
              is the sample size. :cite:`umarov2008q,bercher2011escort`
              :cite:`chhabra1989direct,beck1995thermodynamics`
            * False : it will compute the classical probability, e.g. the ratio
              of empirical frequencies over total number of observation
              :cite:`kolomogoroff2013grundbegriffe`

    supervised_mode : bool | False
        If True it will consider the lavelvector for supervised learning;
        if False unsupervised mode
    forward_computation_mode : bool | False
        Choose if the computation should be forward :
            * True : it will compute joint entropies on the simplicial lattice
              from low dimension to high dimension (co-homological way). For
              each element of the lattice of random-variable the corresponding
              joint probability is estimated. This allows to explore only the
              first low dimensions-rank of the lattice, up to dimension_max
              (in dimension_tot)
            * False : it will compute joint entropies on whole  simplicial
              lattice from high dimension to the marginals (homological way). 
              The joint probability corresponding to all variable is first
              estimated and then projected on lower dimensions using
              conditional rule. This explore the whole lattice, and imposes
              dimension_max = dimension_tot

    nb_bins_histo : int | 200
        Number of values used for entropy and mutual information distribution
        histograms and landscapes.
    p_value_undersampling : float | 0.05
        Real in ]0,1[ value of the probability that a box have a single point
        (e.g. undersampled minimum atomic probability = 1/number of points)
        over all boxes at a given dimension. It provides a confidence to
        estimate the undersampling dimenesion Ku above which information
        etimations shall not be considered.
    compute_shuffle : bool | False
        Choose either :
            * True : it will compute the statictical test of significance of
              the dependencies (pethel et hah 2014) and make shuffles that
              preserve the marginal but the destroys the mutual informations
            * False : no shuffles and test of the mutual information
              estimations is acheived

    p_value : float | 0.05
        Real in ]0,1[ p value of the test of significance of the dependencies
        estimated by mutual info the H0 hypotheis is the mutual Info
        distribution does not differ from the distribution of MI with shuffled
        higher order dependencies
    nb_of_shuffle : int | 20
        Number of shuffles computed
    dim_to_rank : int | 2
        Chosen dimension k to rank the k-tuples as a function information
        functions values.
    number_of_max_val : int | 2
        Number of the first k-tuples with maximum or minimum value to retrieve
        in a dictionary and to plot the corresponding data points k-subspace.
    """
    def __init__(self, dimension_max=16, dimension_tot=16, sample_size=1000,
                 work_on_transpose=False, nb_of_values=9, sampling_mode=1,
                 deformed_probability_mode=False, supervised_mode=False,
                 forward_computation_mode=False, nb_bins_histo=200,
                 p_value_undersampling=0.05, compute_shuffle=False,
                 p_value=0.05, nb_of_shuffle=20, dim_to_rank=2,
                 number_of_max_val=2):
        self.dimension_max = dimension_max
        self.dimension_tot = dimension_tot
        self.sample_size = sample_size
        self.work_on_transpose = work_on_transpose
        self.nb_of_values = nb_of_values
        self.sampling_mode = sampling_mode
        self.deformed_probability_mode = deformed_probability_mode
        self.supervised_mode = supervised_mode
        self.forward_computation_mode = forward_computation_mode
        self.nb_bins_histo  = nb_bins_histo
        self.p_value_undersampling = p_value_undersampling
        self.compute_shuffle = compute_shuffle
        self.p_value = p_value
        self.nb_of_shuffle = nb_of_shuffle
        self.dim_to_rank = dim_to_rank
        self.number_of_max_val = number_of_max_val

    def _validate_parameters(self):
        if self.dimension_max < 2 :
            raise ValueError("dimension_max must be greater than 1")
        if self.dimension_tot < 2 :
            raise ValueError("dimension_tot must be greater than 1")
        if self.sample_size < 2 :
            raise ValueError("sample_size must be greater than 1")
        if self.nb_of_values < 2 :
            raise ValueError("nb_of_values must be greater than 1")
        if self.dimension_max > self.dimension_tot :
            raise ValueError("dimension_tot must be greater or equal than dimension_max")
        if not self.forward_computation_mode :
            if self.dimension_max != self.dimension_tot:
                raise ValueError("if forward_computation_mode then dimension_max must be equal to dimension_tot")
        if self.nb_bins_histo < 2 :
            raise ValueError("nb_bins_histo must be greater than 1")
        if self.p_value > 1 :
            raise ValueError("p_value must be in between 0 and 1")
        if  0 > self.p_value :
            raise ValueError("p_value must be in between 0 and 1")
        if self.p_value_undersampling > 1 :
            raise ValueError("self.p_value_undersampling must be in between 0 and 1")
        if  0 > self.p_value_undersampling :
            raise ValueError("self.p_value_undersampling must be in between 0 and 1")
        if not self.compute_shuffle:
            self.nb_of_shuffle = 0
        if self.dim_to_rank >= self.dimension_max :
            raise ValueError("dim_to_rank must be smaller than dimension_max")



################################################################
#########                 resample                    ##########
#########               DATA MATRIX                   ##########
################################################################
    """
Resample the imput data to nb_of_values for each variables-dimension
nb_of_values is also called the size of the alphabet of the random variable or support
there are 3 different mode of sampling depending on sampling_mode
sampling_mode : (integer: 1,2,3)
                        sampling_mode = 1: normalization taking the max and min of each rows (normaization row by row)
                        sampling_mode = 2: normalization taking the max and min of the whole matrix
TO BE DONE: use panda dataframe .resample to do it...
    """

    def _resample_matrix(self, data_matrix):
        if self.work_on_transpose:
            data_matrix = data_matrix.transpose()
    # find the Min and the Max of the matrix:
        if self.sampling_mode == 1:
            min_matrix = np.min(data_matrix, axis=0)
            max_matrix = np.max(data_matrix, axis=0)
        elif self.sampling_mode == 2:
            min_matrix = np.min(data_matrix)
            max_matrix = np.max(data_matrix)
    #create the amplitude matrix
        ampl_matrix = max_matrix - min_matrix
    #WE RESCALE THE MATRICE AND SAMPLE IT into  nb_of_values #
        data_matrix = np.ceil(((data_matrix-min_matrix)*(self.nb_of_values-1))/(ampl_matrix)).astype(int)
        return data_matrix


################################################################
#########                 compute                     ##########
#########         probability distributions           ##########
################################################################
    """
compute the joint probability distribution of all variables
To avoid to have to explore all  possible  probability (sparse data)
we encode probability as dictionanry, each existing probability has a key

TO DO: import the new simpler function that compute probability and compare
    """

    def _compute_probability(self, data_matrix):
        probability={}
        # in case the data_matrix has a single variable-dimension reshape the vector to matrix
        if len(data_matrix.shape)==1 :
            data_matrix=np.reshape(data_matrix,(data_matrix.shape[0],1))
        for row in range(data_matrix.shape[0]):
            x=''
            for col in range(0,data_matrix.shape[1]):
                x= x+str(int((data_matrix[row,col])))
            probability[x]=probability.get(x,0)+1
        Nbtot=0
        for i in probability.items():
            Nbtot=Nbtot+i[1]
        for i,j in probability.items():
               probability[i]=j/float(Nbtot)
        return probability

###########################################################################################################################
#########          COMPUTE DEFORMED PROBABILITY            ##########
####          AT ALL ORDERS On SET OF SUBSETS           #########
############################################################
    """
    compute the "escort distribution" also called the "deformed probabilities".
    p(n,k)= p(k)^n/ (sum(i)p(i)^n   , where n is the sample size.
    [1] Umarov, S., Tsallis C. and Steinberg S., On a q-Central Limit Theorem Consistent with Nonextensive Statistical Mechanics, Milan j. math. 76 (2008), 307–328
    [2] Bercher,  Escort entropies and divergences and related canonical distribution. Physics Letters A Volume 375, Issue 33, 1 August 2011, Pages 2969-2973
    [3] A. Chhabra, R. V. Jensen, Direct determination of the f(α) singularity spectrum.  Phys. Rev. Lett. 62 (1989) 1327.
    [4] C. Beck, F. Schloegl, Thermodynamics of Chaotic Systems, Cambridge University Press, 1993.
    [5] Zhang, Z., Generalized Mutual Information.  July 11, 2019
TO DO: import the new simpler function that compute probability and compare and use optimal power computation:
https://stackoverflow.com/questions/101439/the-most-efficient-way-to-implement-an-integer-based-power-function-powint-int
    """


    def _compute_deformed_probability(self, data_matrix):
        probability={}
        # in case the data_matrix has a single variable-dimension reshape the vector to matrix
        if len(data_matrix.shape)==1 :
            data_matrix=np.reshape(data_matrix,(data_matrix.shape[0],1))
        sample_size_data= data_matrix.shape[0]
        for row in range(data_matrix.shape[0]):
            x=''
            for col in range(0,data_matrix.shape[1]):
                x= x+str(int((data_matrix[row,col])))
            probability[x]=probability.get(x,0)+1
        Nbtot=0
        for i in probability.items():
            Nbtot=Nbtot+i[1]
        for i,j in probability.items():
               probability[i]=j/float(Nbtot)
        Nbtot_bis=0
        sum_prob=0
        for i in probability.items():
            Nbtot_bis=Nbtot_bis+(i[1]**sample_size_data)
        for i,j in probability.items():
               probability[i]=(j**sample_size_data)/(Nbtot_bis)
               print("probability[i]",probability[i])
               sum_prob=sum_prob+probability[i]
        print("sum_prob",sum_prob)
        return probability

# ###############################################################
# ########          SOME FUNCTIONS USEFULLS            ##########
# ###          AT ALL ORDERS On SET OF SUBSETS          #########
# ###############################################################

    # Entropy Fonction
    def _information(self, x):
        return -x*math.log(x)/math.log(2)

    # Fonction factorielle
    def _factorial(self, x):
        if x < 2:
            return 1
        else:
            return x * self._factorial(x-1)

    # Fonction coeficient binomial (nombre de combinaison de k elements dans [1,..,n])

    def _binomial(self, n,k):
        return self._factorial(n)/(self._factorial(k)*self._factorial(n-k))


#############################################################################
# Fonction _decode(x,n,k,combinat)
#--> renvoie la combinatoire combinat de k variables dans n codée par x
# dans combinat
#les combinaisons de k élements dans [1,..,n] sont en bijection avec
# les entiers x de [0,...,n!/(k!(n-k)!)-1]
# attention numerotation part de 0
#############################################################################

    def _decode(self, x,n,k,combinat):
        if x<0 or n<=0 or k<=0:
            return
        b= self._binomial(n-1,k-1)
        if x<b:
            self._decode(x,n-1,k-1,combinat)
            combinat.append(n)
        else:
            self._decode(x-b,n-1,k,combinat)

#############################################################################
# Fonction _decode_all(x,n,k,combinat)
#--> renvoie la combinatoire (combinat) et l'ordre k associé au code x
# x varie de 0 à (2^n)-1, les n premiers x code pour 1 parmis n
# les suivants codent pour 2 parmis n
# etc... jusquà x=(2^n)-1 qui code pour n parmis n
#les combinaisons de k élements dans [1,..,n] sont en bijection avec
# les entiers x de [0,...,n!/(k!(n-k)!)-1]
#############################################################################

    def _decode_all(self, x, n, order, combinat):
        sumtot=n
        order=1
        Code_order=x
        while x>=sumtot:
            order=order+1
            sumtot=sumtot+ self._binomial(n,order)
            Code_order=Code_order-self._binomial(n,order-1)
        k=order
        self._decode(Code_order,n,k,combinat)


# ##################################################################################
# ###############    COMPUTE ENTROPY                         #######################
# ##################################################################################

    def _compute_entropy(self, probability):
        ntuple=[]
        ntuple1_input=[]
        Nentropie={}
        self._decode(0,self.dimension_tot,self.dimension_max,ntuple1_input)
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        logger = logging.getLogger("compute Proba-Entropy")
        print("Percent of tuples processed : 0")
        for x in range(0,(2**self.dimension_max)-1):
            if self.dimension_max> 10 :
                 if (x) % int(pow(2,self.dimension_max) / 100) == 0:
                     logger.info("PROGRESS: at percent #%i"  % (100*x/pow(2,self.dimension_max)))
            ntuple=[]
            orderInf=0
            self._decode_all(x,self.dimension_max,orderInf,ntuple)
            tuple_code=()
            probability2={}
            for z in range(0,len(ntuple)):
                concat=()
                concat=(ntuple1_input[ntuple[z]-1],)
                tuple_code=tuple_code+concat
            for x,_ in probability.items():
                Codeproba=''
                length=0
                for w in range(1,self.dimension_max+1):
                    if ntuple[length]!=w:
                        Codeproba=Codeproba+'0'
                    else:
                        Codeproba=Codeproba+x[ntuple[length]-1:ntuple[length]]
                        if length<(len(ntuple)-1):
                            length=length+1
                probability2[Codeproba]=probability2.get(Codeproba,0)+probability.get(x,0)
            Nentropie[tuple_code]=0
# to change: the program computes too many times the entropy:  m*2**n instead of
            for x,y in probability2.items():
                Nentropie[tuple_code]=Nentropie.get(tuple_code,0) + self._information(probability2[x])
            probability2={}
            probability2.clear()
            del(probability2)
        return (Nentropie)


###################################################################################
################  COMPUTE FORWARD-CO PROBABILITY AND ENTROPIES  ###################
###################################################################################



    def _compute_forward_entropies(self, data_matrix):
        Nentropie={}
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        logger = logging.getLogger("compute Proba-Entropy")
        print("Percent of tuples processed : 0")
        ################  Create the list of all subsets of i elements in n=dim_tot for all i<dimension_max+1
        allsubsets = lambda n: list(chain(*[combinations(range(1,n), ni) for ni in range(self.dimension_max+1)]))
        list_tuples=allsubsets(self.dimension_tot+1)
        del list_tuples[0]
        ################  Count the number all subsets of i elements in n=dim_tot for all i<dimension_max+1
        if self.dimension_max != self.dimension_tot :
            tot_numb=0
            for  xxx in range(1,self.dimension_max+1):
                tot_numb=tot_numb + self._binomial(self.dimension_tot,xxx)
        counter=0
        for tuple_var in list_tuples:
            ################  create a counter to display the advancement of the script (this is the computationaly costly part)
            counter=counter+1
            if self.dimension_max == self.dimension_tot:
                if counter % int(pow(2, self.dimension_max) / 100) == 0:
                    logger.info("PROGRESS: at percent #%i"  % (100*counter/pow(2,self.dimension_max)))
            else:
                if counter % int(tot_numb / 100) == 0:
                    logger.info("PROGRESS: at percent #%i"  % (100*counter/tot_numb))
            ################  create a sub-matrix of data input for all subsets of variables
            for x in range(0,len(tuple_var)):
                if x==0:
                    matrix_temp = np.reshape(data_matrix[:,tuple_var[x]-1],(data_matrix[:,tuple_var[x]-1].shape[0],1))
                else:
                    matrix_temp=np.concatenate((matrix_temp,np.reshape(data_matrix[:,tuple_var[x]-1],(data_matrix[:,tuple_var[x]-1].shape[0],1))),axis=1)
            ################  compute probability and entropy for each submatrix
            if self.deformed_probability_mode:
                probability =self._compute_deformed_probability(matrix_temp)
            else:
                probability = self._compute_probability(matrix_temp)
            for x,y in probability.items():
                Nentropie[tuple_var]=Nentropie.get(tuple_var,0)+ self._information(probability[x])
        return  Nentropie




    def simplicial_entropies_decomposition(self, data_matrix) :
        self._validate_parameters()
        data_matrix = self._resample_matrix(data_matrix)
        if self.forward_computation_mode:
            Nentropie = self._compute_forward_entropies(data_matrix)
        else:
            if self.deformed_probability_mode:
                probability =self._compute_deformed_probability(data_matrix)
            else:
                probability = self._compute_probability(data_matrix)
            Nentropie = self._compute_entropy(probability)
        return Nentropie



##############################################################################
## Function binomial_subgroups COMBINAT Gives all binomial k subgroup of a group
##############################################################################


    def simplicial_infomut_decomposition(self, Nentropie_input):
        Ninfomut={}
        for x,y in Nentropie_input.items():
            for k in range(1, len(x)+1):
                for subset in itertools.combinations(x, k):
                    Ninfomut[x]=Ninfomut.get(x,0)+ ((-1)**(len(subset)+1))*Nentropie_input[subset]
        return (Ninfomut)

#########################################################################
#########################################################################
######             Histogramms ENTROPY          #########################
######           &  ENTROPY LANDSCAPES          #########################
######                FIGURE 4                  #########################
#########################################################################
#########################################################################

    def entropy_simplicial_lanscape(self, Nentropie):
        num_fig = 1
        plt.figure(num_fig,figsize=(18,10))
        moyenne={}
        nbpoint={}
        matrix_distrib_info=np.array([])
        maxima_tot=-1000000.00
        minima_tot=1000000.00
        ListEntropyordre={}
        undersampling_percent=np.array([])


        for i in range(1,self.dimension_max+1):
            ListEntropyordre[i]=[]

        for x,y in Nentropie.items():
            ListEntropyordre[len(x)].append(y)
            moyenne[len(x)]=moyenne.get(len(x),0)+y
            nbpoint[len(x)]=nbpoint.get(len(x),0)+1
            if y>maxima_tot:
                maxima_tot=y
            if y<minima_tot:
                minima_tot=y
        delta_entropy_histo = (maxima_tot-minima_tot)/self.nb_bins_histo

        for a in range(1,self.dimension_max+1):
            if self.dimension_max<=9 :
                plt.subplot(3,3,a)
            else :
                if self.dimension_max<=16 :
                    plt.subplot(4,4,a)
                else :
                    if self.dimension_max<=20 :
                        plt.subplot(5,4,a)
                    else :
                        plt.subplot(5,5,a)
 # compute the Ku undersampling bound
            nb_undersampling_point=0
            for x in range(0,len(ListEntropyordre[a])):
 #               if ListEntropyordre[a][x] >= ((math.log(self.sample_size)/math.log(2))-0.00000001):
                if ListEntropyordre[a][x] >= ((math.log(self.sample_size)/math.log(2))-delta_entropy_histo):
                    nb_undersampling_point=nb_undersampling_point+1
                    if a ==1:
                        print(ListEntropyordre[a][x])
            percent_undersampled =  100*nb_undersampling_point/self._binomial(self.dimension_max,a)
            undersampling_percent = np.hstack((undersampling_percent, percent_undersampled))
            print('undersampling percent in dim ',a,' = ', percent_undersampled )
            ListEntropyordre[a].append(minima_tot-0.1)
            ListEntropyordre[a].append(maxima_tot+0.1)
            n, bins, patches = plt.hist(ListEntropyordre[a], self.nb_bins_histo, facecolor='g')
            plt.axis([minima_tot, maxima_tot,0,n.max()])
            plt.title(str('H'+str(a)+' dist'))
            if a==1 :
                matrix_distrib_info=n
            else:
                matrix_distrib_info=np.c_[matrix_distrib_info,n]
            plt.grid(True)

        boolean_test = True
        undersampling_dim = self.dimension_max
        for xxxx in  range(0,self.dimension_max) :
            if undersampling_percent[xxxx] > (self.p_value_undersampling*100) and boolean_test:
                undersampling_dim = xxxx+1
                boolean_test = False
        print('the undersampling dimension is ', undersampling_dim, 'with self.p_value_undersampling',self.p_value_undersampling)

        num_fig=num_fig+1
        plt.figure(num_fig)
        abssice_degree=np.linspace(1, self.dimension_max, self.dimension_max)
        plt.plot(abssice_degree,undersampling_percent)
        plt.ylabel('percent of undersampled points')
        plt.title(str('Undersampling dimension (p>'+str(self.p_value_undersampling)+'), ku='+str(undersampling_dim)))
        plt.xlabel('dimension')
        plt.grid(True)

        num_fig=num_fig+1
        fig_entropylandscape =plt.figure(num_fig,figsize=(18, 10))
        matrix_distrib_info=np.flipud(matrix_distrib_info)
        plt.matshow(matrix_distrib_info, cmap='jet', aspect='auto', extent=[0,self.dimension_max,minima_tot-0.1,maxima_tot+0.1], norm=LogNorm(), fignum= num_fig)
        plt.axis([0,self.dimension_max,minima_tot,maxima_tot])
        cbar = plt.colorbar()
        cbar.set_label('# of tuples', rotation=270)
        plt.grid(False)
        plt.xlabel('dimension')
        plt.ylabel('Hk value (bits)')
        plt.title('Hk landscape')
        fig_entropylandscape.set_size_inches(18, 10)
        plt.show()


#########################################################################
#########################################################################
######      Histogramms MUTUAL INFORMATION      #########################
######           &  INFOMUT LANDSCAPES          #########################
#########################################################################
#########################################################################
    def mutual_info_simplicial_lanscape(self, Ninfomut) :
        num_fig = 1
        matrix_distrib_infomut=np.array([])
#######################################################
#   COMPUTE THE LIST OF INFOMUT VALUES FOR EACH DEGREE
# Display every Histo with its own scales:
#######################################################
# Compute the list of Infomut at each degree
        ListInfomutordre={}
        maxima_tot=-1000000.00
        minima_tot=1000000.00
        for i in range(1,self.dimension_max+1):
            ListInfomutordre[i]=[]

        for x,y in Ninfomut.items():
            ListInfomutordre[len(x)].append(y)
            if y>maxima_tot:
                maxima_tot=y
            if y<minima_tot:
                minima_tot=y

# Compute the list of Infomut at each degree for each SHUFFLE and sums the distributions
# (we sum the districbution because the original n-shuffles vectors would be too big )
        if self.compute_shuffle == True:
            hist_sum_SHUFFLE = {}
            for a in range(1,self.dimension_max+1):
                hist_sum_SHUFFLE[a]=[]

            for k in range(self.nb_of_shuffle):
                print('k=',k)
                name_object= 'INFOMUT'+str(k)
                Ninfomut = load_obj(name_object)
                ListInfomutordreSHUFFLE={}
                for i in range(1,self.dimension_max+1):
                    ListInfomutordreSHUFFLE[i]=[]
                for x,y in Ninfomut.items():
                    ListInfomutordreSHUFFLE[len(x)].append(y)
                    if y>maxima_tot:
                        maxima_tot=y
                    if y<minima_tot:
                        minima_tot=y
                for i in range(1,self.dimension_max+1):
                    nSHUFFLE,bin_edgesSHUFFLE  = np.histogram(ListInfomutordreSHUFFLE[i], self.nb_bins_histo, (minima_tot,maxima_tot))
                    if k == 0 :
                        hist_sum_SHUFFLE[i] = nSHUFFLE
                    else:
                        hist_sum_SHUFFLE[i] = np.sum([hist_sum_SHUFFLE[i],nSHUFFLE],axis=0)
                ListInfomutordreSHUFFLE.clear()
                del(ListInfomutordreSHUFFLE)
            for i in range(1,self.dimension_max+1):
                hist_sum_SHUFFLE[i]=np.concatenate([[0],hist_sum_SHUFFLE[i]])
                if self.dimension_max<9 :
                    plt.subplot(3,3,i)
                else :
                    if self.dimension_max<=16 :
                        plt.subplot(4,4,i)
                    else :
                        if self.dimension_max<=20 :
                            plt.subplot(5,4,i)
                        else :
                            plt.subplot(5,5,i)
                plt.plot(bin_edgesSHUFFLE,hist_sum_SHUFFLE[i])
                plt.axis([minima_tot, maxima_tot,0,hist_sum_SHUFFLE[i].max()])
                num_fig=num_fig+1
#   COMPUTE THE HISTOGRAMS OF THE LIST OF INFOMUT VALUES FOR EACH DEGREE
#   If shuffle is true it also compute the signifiance test against independence null hypothesis

        fig_Histo_infomut = plt.figure(num_fig,figsize=(18,10))
        if self.compute_shuffle == True:
            low_signif_bound={}
            high_signif_bound={}
            lign_signif={}
        for a in range(1,self.dimension_max+1):
            if self.dimension_max<9 :
                plt.subplot(3,3,a)
            else :
                if self.dimension_max<=16 :
                    plt.subplot(4,4,a)
                else :
                    if self.dimension_max<=20 :
                        plt.subplot(5,4,a)
                    else :
                        plt.subplot(5,5,a)
            ListInfomutordre[a].append(minima_tot-0.1)
            ListInfomutordre[a].append(maxima_tot+0.1)
            n, bins, patches = plt.hist(ListInfomutordre[a], self.nb_bins_histo, facecolor='r')
            if self.compute_shuffle == True:
                plt.plot(bin_edgesSHUFFLE,hist_sum_SHUFFLE[a]*(1/self.nb_of_shuffle),color="blue")
                cumul=0
                first_signif = True
                second_signif = False
                lign_signif[a]=[]
                lign_signif[a] = np.zeros_like(hist_sum_SHUFFLE[a])
                for x in range(0,len(hist_sum_SHUFFLE[a])):
                    cumul=cumul+hist_sum_SHUFFLE[a][x]
                    if first_signif == True:
                        if cumul >= (binomial(self.dimension_max,a)*self.nb_of_shuffle*self.p_value):
                            low_signif_bound[a] = bin_edgesSHUFFLE[x]
                            lign_signif[a][x] =hist_sum_SHUFFLE[a].max()
                            first_signif = False
                            second_signif = True
                            print('low_signif_bound in dim',a, ' = ',  low_signif_bound[a])
                    if second_signif == True:
                        if cumul >= (binomial(self.dimension_max,a)*self.nb_of_shuffle*(1-self.p_value)):
                            high_signif_bound[a] = bin_edgesSHUFFLE[x]
                            lign_signif[a][x] =hist_sum_SHUFFLE[a].max()
                            second_signif = False
                            print('high_signif_bound in dim',a, ' = ',  high_signif_bound[a])
                nb_of_signif_low = 0
                nb_of_signif_high = 0
                for x in range(0,len(ListInfomutordre[a])):
                    if ListInfomutordre[a][x] <= low_signif_bound[a] :
                        nb_of_signif_low = nb_of_signif_low+1
                    if ListInfomutordre[a][x] >= high_signif_bound[a] :
                        nb_of_signif_high = nb_of_signif_high+1
                print('nb_of_signif_low in dim',a, ' = ',  nb_of_signif_low)
                print('nb_of_signif_high in dim',a, ' = ',  nb_of_signif_high)
                plt.plot(bin_edgesSHUFFLE,lign_signif[a]*(1/self.nb_of_shuffle),color="green")
            plt.axis([minima_tot, maxima_tot,0,n.max()])
            plt.title(str('I'+str(a)+' dist'))
            if a==1 :
                matrix_distrib_infomut=n
            else:
                matrix_distrib_infomut=np.c_[matrix_distrib_infomut,n]
            plt.grid(True)

#   COMPUTE THE INFOMUT LANDSCAPE FROM THE HISTOGRAMS AS THE MATRIX  matrix_distrib_infomut
#   If shuffle is true it also plots the signifiance test against independence null hypothesis
        num_fig=num_fig+1
        fig_infolandscape =plt.figure(num_fig,figsize=(18, 10))
        matrix_distrib_infomut=np.flipud(matrix_distrib_infomut)
        plt.matshow(matrix_distrib_infomut, cmap='jet', aspect='auto', extent=[0,self.dimension_max,minima_tot-0.1,maxima_tot+0.1], norm=LogNorm(), fignum= num_fig)
        plt.axis([0,self.dimension_max,minima_tot,maxima_tot])
        if self.compute_shuffle == True:
            abssice=np.linspace(0.5, self.dimension_max-0.5, self.dimension_max)
            low_ordinate=[]
            high_ordinate=[]
            for a in range(1,self.dimension_max+1):
                low_ordinate.append(low_signif_bound[a])
                high_ordinate.append(high_signif_bound[a])
            plt.plot(abssice, low_ordinate, marker='o', color='black')
            plt.plot(abssice, high_ordinate, marker='o',color='black')
        plt.title('Ik landscape')
        plt.xlabel('dimension')
        plt.ylabel('Ik value (bits)')
        fig_infolandscape.set_size_inches(18, 10)
        cbar = plt.colorbar()
        cbar.set_label('# of tuples', rotation=270)
        plt.grid(False)
        plt.show()


#########################################################################
#########################################################################
######    CONDITIONAL ENTROPY and MUTUAL INFORMATION    #################
#########################################################################
#########################################################################
    """
    This function computes all conditional entropy and conditional informations (conditionning by a single variable)
    They are given by chain rules and correspond to each edges of the lattice.
    the output is a list of dictionaries dico_input_CONDtot[i-1] items are of the forms ((5, 7, 9), 0.3528757654347521)  for
    the information of 5,7 knowing 9, e.g. I(5,7|9)
    """

    def conditional_info_simplicial_lanscape(self, dico_input):
        num_fig = 1
        fig_Histo_infomut = plt.figure(num_fig,figsize=(18,10))
        matrix_distrib_info=np.array([])
        # Display every Histo with its own scales:
        ListInfomutcond={}
        maxima_tot=-1000000.00
        minima_tot=1000000.00
        dico_input_COND=[]
        dico_input_CONDtot=[]
        for i in range(1,self.dimension_max+1):
            ListInfomutcond[i]=[]
            dico_input_CONDperORDER=[]
            dico_input_COND.append(dico_input_CONDperORDER)
            dicobis={}
            dico_input_CONDtot.append(dicobis)
            for j in range(1,self.dimension_max+1):
                dico={}
                dico_input_COND[i-1].append(dico)

        for i in range(1,self.dimension_max+1):
            for x,y in dico_input.items():
                if len(x)>1:
                    for b in x:
                        if (b==i):
                            xbis= tuple(a for a in x if (a!=i))
                            cond= dico_input[xbis]-y
                            if cond>maxima_tot:
                                maxima_tot=cond
                            if cond<minima_tot:
                               minima_tot=cond
     # for conditioning per degree
                            ListInfomutcond[len(x)-1].append(cond)
     # for conditioning per variable
                            dico_input_COND[len(x)-1][i-1][xbis]=cond
                            xter = xbis + ((i),)
                            dico_input_CONDtot[len(x)-1][xter]=cond
     # The last term in the tuple is the conditionning variable
        for a in range(1,self.dimension_max+1):
            if self.dimension_max<9 :
                plt.subplot(3,3,a)
            else :
                if self.dimension_max<=16 :
                    plt.subplot(4,4,a)
                else :
                    if self.dimension_max<=20 :
                        plt.subplot(5,4,a)
                    else :
                        plt.subplot(5,5,a)
            ListInfomutcond[a].append(minima_tot-0.1)
            ListInfomutcond[a].append(maxima_tot+0.1)
            n, bins, patches = plt.hist(ListInfomutcond[a], self.nb_bins_histo, facecolor='r')
            plt.title(str('condInfo'+str(a)+' dist'))
            plt.axis([minima_tot, maxima_tot,0,n.max()])
            plt.grid(True)
            if a==1 :
               matrix_distrib_info=n
            else:
               matrix_distrib_info=np.c_[matrix_distrib_info,n]

        num_fig=num_fig+1
        fig_infocondlandscape =plt.figure(num_fig)
        matrix_distrib_info=np.flipud(matrix_distrib_info)
        plt.matshow(matrix_distrib_info, cmap='jet', aspect='auto', extent=[0,self.dimension_max,minima_tot-0.1,maxima_tot+0.1], norm=LogNorm(), fignum= num_fig)
        plt.axis([0,self.dimension_max,minima_tot,maxima_tot])
        cbar = plt.colorbar()
        cbar.set_label('# of tuples', rotation=270)
        plt.title('condInfo landscape')
        plt.xlabel('dimension')
        plt.ylabel('condInfo value (bits)')
        fig_infocondlandscape.set_size_inches(18, 10)
        plt.grid(False)

        plt.show()
        return dico_input_CONDtot


#########################################################################
#########################################################################
######      Histogramms ENTROPY & MUTUAL INFORMATION   ##################
###### computes entropy vs information for each degree ##################
######       ENTROPY  &  INFOMUT LANDSCAPES         #####################
######                FIGURE 6                  #########################
#########################################################################
#########################################################################
### ENTROPY VS ENERGY VS VOL  Willard Gibbs' 1873 figures two and three
# (above left and middle) used by Scottish physicist James Clerk Maxwell
# in 1874 to create a three-dimensional entropy (x), volume (y), energy (z)
# thermodynamic surface diagram

    def display_entropy_energy_landscape(self, Ninfomut, Nentropie):

        ListInfomutordre={}
        ListEntropyordre={}
        maxima_tot=-1000000.00
        minima_tot=1000000.00
        for i in range(1,self.dimension_max+1):
            ListInfomutordre[i]=[]
            ListEntropyordre[i]=[]

        for x,y in Ninfomut.items():
            ListInfomutordre[len(x)].append(y)
            ListEntropyordre[len(x)].append(Nentropie.get(x))
            if y>maxima_tot:
                maxima_tot=y
            if y<minima_tot:
                minima_tot=y
        for a in range(1,self.dimension_max+1):
            ListInfomutordre[a].append(minima_tot-0.1)
            ListInfomutordre[a].append(maxima_tot+0.1)
        num_fig = 1
        num_fig = num_fig+1
        fig_entropy_eneregy =plt.figure(num_fig)

        maxima_tot_entropy=-1000000.00
        minima_tot_entropy=1000000.00
        for x,y in Nentropie.items():
            if y>maxima_tot_entropy:
                maxima_tot_entropy=y
            if y<minima_tot_entropy:
                minima_tot_entropy=y
        for a in range(1,self.dimension_max+1):
            if self.dimension_max<=9 :
                plt.subplot(3,3,a)
            else :
                if self.dimension_max<=16 :
                    plt.subplot(4,4,a)
                else :
                    if self.dimension_max<=20 :
                        plt.subplot(5,4,a)
                    else :
                        plt.subplot(5,5,a)
            ListEntropyordre[a].append(minima_tot_entropy-0.1)
            ListEntropyordre[a].append(maxima_tot_entropy+0.1)
            plt.hist2d(ListEntropyordre[a], ListInfomutordre[a], bins=int(self.nb_bins_histo/2), norm=LogNorm())
            plt.title(str('dim '+str(a)))
            cbar = plt.colorbar()
            cbar.ax.set_ylabel('Counts')
            plt.axis([minima_tot_entropy, maxima_tot_entropy,minima_tot,maxima_tot])
        fig_entropy_eneregy.suptitle('Entropy vs. Energy landsacpe', fontsize=16)
        fig_entropy_eneregy.set_size_inches(18, 10)
        plt.show()


#########################################################################
#########################################################################
######    TOTAL CORRELATION - INTEGRATED INFORMATIOn    #################
######                    FREE ENERGY                   #################
#########################################################################
#########################################################################
    """
    This function computes all total correlations or integrated information or free energy
    """

    def total_correlation_simplicial_lanscape(self, Nentropie):
        num_fig = 1
        plt.figure(num_fig,figsize=(18,10))
        matrix_distrib_info=np.array([])
        maxima_tot=-1000000.00
        minima_tot=1000000.00
        list_tot_correlation={}
        Ntotal_correlation={}

        for i in range(1,self.dimension_max+1):
            list_tot_correlation[i]=[]

        for x,y in Nentropie.items():
            sum_marginals = 0
            for var in x:
                sum_marginals = sum_marginals + Nentropie[(var,)]
            total_corr =   sum_marginals - y
            Ntotal_correlation.update( {x : total_corr} )
            list_tot_correlation[len(x)].append(total_corr)
            if total_corr>maxima_tot:
                maxima_tot=total_corr
            if total_corr<minima_tot:
                minima_tot=total_corr

        for a in range(1,self.dimension_max+1):
            if self.dimension_max<=9 :
                plt.subplot(3,3,a)
            else :
                if self.dimension_max<=16 :
                    plt.subplot(4,4,a)
                else :
                    if self.dimension_max<=20 :
                        plt.subplot(5,4,a)
                    else :
                        plt.subplot(5,5,a)
            list_tot_correlation[a].append(minima_tot-0.1)
            list_tot_correlation[a].append(maxima_tot+0.1)
            n, bins, patches = plt.hist(list_tot_correlation[a], self.nb_bins_histo, facecolor='b')
            plt.axis([minima_tot, maxima_tot,0,n.max()])
            plt.title(str('G'+str(a)+' dist'))
            if a==1 :
                matrix_distrib_info=n
            else:
                matrix_distrib_info=np.c_[matrix_distrib_info,n]
            plt.grid(True)

        num_fig=num_fig+1
        fig_total_correlation_landscape =plt.figure(num_fig,figsize=(18, 10))
        matrix_distrib_info=np.flipud(matrix_distrib_info)
        plt.matshow(matrix_distrib_info, cmap='jet', aspect='auto', extent=[0,self.dimension_max,minima_tot-0.1,maxima_tot+0.1], norm=LogNorm(), fignum= num_fig)
        plt.axis([0,self.dimension_max,minima_tot,maxima_tot])
        cbar = plt.colorbar()
        cbar.set_label('# of tuples', rotation=270)
        plt.grid(False)
        plt.xlabel('dimension')
        plt.ylabel('Gk value (bits)')
        plt.title('Total correlation Gk landscape')
        fig_total_correlation_landscape.set_size_inches(18, 10)
        plt.show()
        return Ntotal_correlation

#########################################################################
#########################################################################
######          INFORMATION DISTANCE AND VOLUMES        #################
#########################################################################
#########################################################################
    """
    This function computes all Information distance V(X,Y)=H(X,Y)-I(X,Y), a 2-volume and its generalization to k-volume: Vk=Hk-Ik for all the simplicial structure.
    """

    def information_volume_simplicial_lanscape(self, Nentropie, Ninfomut):
        num_fig = 1
        plt.figure(num_fig,figsize=(18,10))
        matrix_distrib_info=np.array([])
        maxima_tot=-1000000.00
        minima_tot=1000000.00
        list_info_volume={}
        Ninfo_volume={}

        for i in range(1,self.dimension_max+1):
            list_info_volume[i]=[]

        for x,y in Nentropie.items():
            sum_marginals = 0
            info_vol =  y - Ninfomut[x]
            Ninfo_volume.update( {x : info_vol} )
            list_info_volume[len(x)].append(info_vol)
            if info_vol>maxima_tot:
                maxima_tot=info_vol
            if info_vol<minima_tot:
                minima_tot=info_vol

        for a in range(1,self.dimension_max+1):
            if self.dimension_max<=9 :
                plt.subplot(3,3,a)
            else :
                if self.dimension_max<=16 :
                    plt.subplot(4,4,a)
                else :
                    if self.dimension_max<=20 :
                        plt.subplot(5,4,a)
                    else :
                        plt.subplot(5,5,a)
            list_info_volume[a].append(minima_tot-0.1)
            list_info_volume[a].append(maxima_tot+0.1)
            n, bins, patches = plt.hist(list_info_volume[a], self.nb_bins_histo, facecolor='b')
            plt.axis([minima_tot, maxima_tot,0,n.max()])
            plt.title(str('V'+str(a)+' dist'))
            if a==1 :
                matrix_distrib_info=n
            else:
                matrix_distrib_info=np.c_[matrix_distrib_info,n]
            plt.grid(True)

        num_fig=num_fig+1
        fig_total_correlation_landscape =plt.figure(num_fig,figsize=(18, 10))
        matrix_distrib_info=np.flipud(matrix_distrib_info)
        plt.matshow(matrix_distrib_info, cmap='jet', aspect='auto', extent=[0,self.dimension_max,minima_tot-0.1,maxima_tot+0.1], norm=LogNorm(), fignum= num_fig)
        plt.axis([0,self.dimension_max,minima_tot,maxima_tot])
        cbar = plt.colorbar()
        cbar.set_label('# of tuples', rotation=270)
        plt.grid(False)
        plt.xlabel('dimension')
        plt.ylabel('Vk value (bits)')
        plt.title('Information distance and Volume Vk landscape')
        fig_total_correlation_landscape.set_size_inches(18, 10)
        plt.show()
        return Ninfo_volume

# ##########################################################################################
# ###############              RANKING and DISPLAY of the             ######################
# ###############   n first higher and lower conditional information  ######################
# ##########################################################################################
    """
    This function prints all the conditional information at a given dimension-order (dimension 1 for H(Xi|Xj) dimension 2 for H(Xi,Xk|Xj)...)
    the dico_input_CONDtot[i-1] items are of the forms ((5, 7, 9), 0.352875765,347521) for the information of 5,7 knowing 9, e.g. I(5,7|9)
    """
    def display_higher_lower_cond_information(self, dico_input_CONDtot):

        print('The conditional information at dim',(self.dim_to_rank-1))
        print(OrderedDict(sorted(dico_input_CONDtot[self.dim_to_rank-1].items(), key=lambda t: t[1])))

###############################################################
########          Ring representation               ##########
###                Mutual information               #########
###### http://networkx.readthedocs.org/en/stable/   #########
###############################################################
    """
    A ring network - Graph representation of 2nd order - pairwise mutual information and firts order marginal entropy
    only use the (symmetric nul diagonal - or upper triangular) adjacency matrix of 2-MI
    """

    def mutual_info_pairwise_network(self, Ninfomut) :
        infomut_per_order=[]
        for x in range(self.dimension_max+1):
            info_dicoperoder={}
            infomut_per_order.append(info_dicoperoder)
        for x,y in Ninfomut.items():
            infomut_per_order[len(x)][x]=Ninfomut[x]
        num_fig = 1
        plt.figure(num_fig,figsize=(18, 10))
        netring = nx.Graph()
        list_of_node=[]
        list_of_size=[]
        list_of_edge=[]
        list_of_width=[]
        list_of_labels={}

        for x,y in infomut_per_order[1].items():
            tuple_interim=(x)
            list_of_node.append(x[0])
            list_of_labels[x]=tuple_interim
            number_node=[x]
            code_var=tuple(number_node)
            list_of_size.append((infomut_per_order[1][x]**2) *500)
            netring.add_node(x[0])
        tuple_interim=()
        for x,y in infomut_per_order[2].items():
            list_of_edge.append(x)
            var_1=x[0]
            var_2=x[1]
            netring.add_edge(var_1, var_2, weight= (infomut_per_order[2][x]) )
            list_of_width.append((infomut_per_order[2][x]*10))
        plt.subplot(1, 2, 1)
        nx.draw_circular(netring, with_labels= True, nodelist = list_of_node,edgelist = list_of_edge, width= list_of_width, node_size = list_of_size)
        adjacency_matrix = np.zeros((len(list_of_node), len(list_of_node)))
        for x,y in infomut_per_order[2].items():
            adjacency_matrix[x[0]-1,x[1]-1] = y
            adjacency_matrix[x[1]-1,x[0]-1] = y
        for x,y in infomut_per_order[1].items():
            adjacency_matrix[x[0]-1,x[0]-1] = infomut_per_order[1][(x[0],)]
        plt.subplot(1, 2, 2)
        plt.title('Information adjacency matrix (I1 and I2)')
        plt.imshow(adjacency_matrix, cmap='hot')
        cbar = plt.colorbar()
        cbar.set_label('Information (bits)', rotation=270)
        plt.show()
        return adjacency_matrix



# ########################################################################################
# ###############              RANKING and DISPLAY of the           ######################
# ###############   n first higher and lower Mutual information     ######################
# ########################################################################################

    '''
    This function ranks the tuples in dimension k=dim_to_rank as a funtion  entropy or information
    and print and plot de data points k subsapce of  the n first maximum and minimum values (n=number_of_max_val)
    if the dimension is 2,3 or 4 (when ploting is possible). For 4D plot the 4th dimension is given by the colorscale
    of the points
    '''
    def display_higher_lower_information(self, dico_input, dataset):
        dico_at_order = {}
        for x,y in dico_input.items():
            if len(x) == self.dim_to_rank :
                dico_at_order[x]=dico_input[x]
        topitems = heapq.nlargest(self.number_of_max_val, dico_at_order.items(), key=itemgetter(1))
        topitemsasdict_max = dict(topitems)

        # here we plot the number_of_max_val first maxima
        fig1 = plt.figure(figsize=(18, 10))
        ax1=np.empty((1,self.number_of_max_val))
        nb_plot = 0
        aaaa=0

        for key in topitemsasdict_max :
            aaaa=aaaa+1
            print(aaaa, "max value in dimension", self.dim_to_rank ," is for the tuple", key,  "   with Fk value  :", topitemsasdict_max[key])

            if self.dim_to_rank == 2:
                ax1 = fig1.add_subplot(1, self.number_of_max_val, nb_plot+1)
                ax1.scatter(dataset[:,key[0]-1], dataset[:,key[1]-1],  c= 'red', marker='8')
                string_title = str(str(aaaa)+"max : F"+str(self.dim_to_rank)+"("+ str(key)+")="+ str(round(topitemsasdict_max[key],2)))
                ax1.set_title(string_title)
                ax1.set_xlabel('variable'+str(key[0]))
                ax1.set_ylabel('variable'+str(key[1]))
                ax1.grid(True)
                nb_plot =nb_plot +1
            elif self.dim_to_rank == 3:
                ax1 = fig1.add_subplot(1, self.number_of_max_val, nb_plot+1, projection='3d')
                ax1.scatter(dataset[:,key[0]-1], dataset[:,key[1]-1], dataset[:,key[2]-1],  c= 'red', marker='8')
                ax1.set_xlabel('variable'+str(key[0]))
                ax1.set_ylabel('variable'+str(key[1]))
                ax1.set_zlabel('variable'+str(key[2]))
                string_title = str(str(aaaa)+"max : F"+str(self.dim_to_rank)+"("+ str(key)+")="+ str(round(topitemsasdict_max[key],2)))
                ax1.set_title(string_title)
                ax1.grid(True)
                nb_plot =nb_plot +1
            elif self.dim_to_rank == 4:
                ax1 = fig1.add_subplot(1, self.number_of_max_val, nb_plot+1, projection='3d')
                pts = ax1.scatter(dataset[:,key[0]-1], dataset[:,key[1]-1], dataset[:,key[2]-1], c=dataset[:,key[3]-1], cmap='jet', alpha=1, marker='8')
                ax1.set_xlabel('variable'+str(key[0]))
                ax1.set_ylabel('variable'+str(key[1]))
                ax1.set_zlabel('variable'+str(key[2]))
                cbar = fig1.colorbar(pts, ax=ax1)
                string_title = str(str(aaaa)+"max : F"+str(self.dim_to_rank)+"("+ str(key)+")="+ str(round(topitemsasdict_max[key],2)))
                ax1.set_title(string_title)
                ax1.grid(True)
                nb_plot =nb_plot +1

        topitems = heapq.nsmallest(self.number_of_max_val, dico_at_order.items(), key=itemgetter(1))
        topitemsasdict_min = dict(topitems)

        # here we plot the number_of_max_val first minima

        fig2 = plt.figure(figsize=(18, 10))
        ax2=np.empty((1,self.number_of_max_val))
        aaaa=0
        nb_plot = 0

        for key in topitemsasdict_min :
            aaaa=aaaa+1
            print(aaaa, "min value in dimension", self.dim_to_rank ," is for the tuple", key,  "   with Fk value  :", topitemsasdict_min[key])
            if self.dim_to_rank == 2:
                ax2 = fig2.add_subplot(1, self.number_of_max_val, nb_plot+1)
                ax2.scatter(dataset[:,key[0]-1], dataset[:,key[1]-1],  c= 'red', marker='8')
                string_title = str(str(aaaa)+"min : F"+str(self.dim_to_rank)+"("+ str(key)+")="+ str(round(topitemsasdict_min[key],2)))
                ax2.set_title(string_title)
                ax2.set_xlabel('variable'+str(key[0]))
                ax2.set_ylabel('variable'+str(key[1]))
                ax2.grid(True)
                nb_plot =nb_plot +1
            elif self.dim_to_rank == 3:
                ax2 = fig2.add_subplot(1, self.number_of_max_val, nb_plot+1, projection='3d')
                ax2.scatter(dataset[:,key[0]-1], dataset[:,key[1]-1], dataset[:,key[2]-1],  c= 'red', marker='8')
                ax2.set_xlabel('variable'+str(key[0]))
                ax2.set_ylabel('variable'+str(key[1]))
                ax2.set_zlabel('variable'+str(key[2]))
                string_title = str(str(aaaa)+"min : F"+str(self.dim_to_rank)+"("+ str(key)+")="+ str(round(topitemsasdict_min[key],2)))
                ax2.set_title(string_title)
                ax2.grid(True)
                nb_plot =nb_plot +1
            elif self.dim_to_rank == 4:
                ax2 = fig2.add_subplot(1, self.number_of_max_val, nb_plot+1, projection='3d')
                pts = ax2.scatter(dataset[:,key[0]-1], dataset[:,key[1]-1], dataset[:,key[2]-1], c=dataset[:,key[3]-1], cmap='jet', alpha=1, marker='8')
                ax2.set_xlabel('variable'+str(key[0]))
                ax2.set_ylabel('variable'+str(key[1]))
                ax2.set_zlabel('variable'+str(key[2]))
                cbar = fig2.colorbar(pts, ax=ax2)
                string_title = str(str(aaaa)+"min : F"+str(self.dim_to_rank)+"("+ str(key)+")="+ str(round(topitemsasdict_min[key],2)))
                ax2.set_title(string_title)
                ax2.grid(True)
                nb_plot =nb_plot +1
        plt.show()
        return (topitemsasdict_max, topitemsasdict_min)


# ############################################################################################
# ###############              PLOT MEAN INFORMATION RATE               ######################
# ###############    PLOT MEAN ENTROPY RATE NORMALISED BY BINOMIAL      ######################
# ############################################################################################
    '''
    This function plotts at each dimension k the mean information (entropy, Mutual-Information...) rate and the mean information
    (entropy, Mutual-Information...) rate normalised by the binomial coeficient C(k,n)
    '''
    def display_mean_information(self, dico_input):
#########################################################################
###########                MEAN INFO                          ###########
###########  SUM INFO  normalised by Binomial coefficients    ###########
###########  (MEAN-FIELD APPROXIMATION HOMOGENEOUS SYSTEM)    ###########
#########################################################################
        info_sum_order={}
        for x,y in dico_input.items():
            info_sum_order[len(x)]=info_sum_order.get(len(x),0)+dico_input[x]
        num_fig = 1
        num_fig = num_fig+1
        plt.figure(num_fig)
        maxordonnee = -1000000.00
        minordonnee = 1000000.00
        mean = []
        xxx = []
        for x,y in info_sum_order.items():
            mean.append(y/self._binomial(self.dimension_tot,x))
            xxx.append(x)
        plt.plot(xxx, mean, linestyle='-', marker='o', color='b', linewidth=2)
        plt.ylabel('(Bits/symbols)')
        plt.title('Mean info function')
        plt.grid(True)

#########################################################################
###########                MEAN INFO  RATE                    ###########
###########  SUM INFO  normalised by Binomial coefficients    ###########
###########  (MEAN-FIELD APPROXIMATION HOMOGENEOUS SYSTEM)    ###########
#########################################################################

        num_fig = num_fig+1
        plt.figure(num_fig)
        maxordonnee=-1000000.00
        minordonnee=1000000.00
        rate = []
        xxx = []
        for x,y in info_sum_order.items():
            rate.append(y/(self._binomial(self.dimension_tot,x)*x))
            xxx.append(x)
        plt.plot(xxx, rate, linestyle='-', marker='o', color='b', linewidth=2)
        plt.ylabel('(Bits/symbols)')
        plt.title('Mean info rate function')
        plt.grid(True)
        plt.show()
        return (mean, rate)


###############################################################
###############################################################
########              INFORMATION FIT                ##########
########                                             ##########
###############################################################
###############################################################
    '''
    This function is just a basic wrapper on previous functions to provide a scikit or tensorflow (...) like fit function
    ... to help users.
    '''
    def fit( self, dataset):
        Nentropie = self.simplicial_entropies_decomposition(dataset)
        Ninfomut = self.simplicial_infomut_decomposition(Nentropie)
        return Ninfomut, Nentropie

###############################################################
###############################################################
########              INFORMATION PATHS              ##########
########  INFORMATION COMPLEX - FREE-ENERGY COMPLEX  ##########
###############################################################
###############################################################
    '''
    This function compute and plotts approximation of the information (free-energy) complex by computing information paths: An information path IPk  of degree k
    on Ik landscape is defined as a sequence of elements of the lattice that begins at the least element of the lattice (the identity-constant “0”),
    travels along edges from element to element of increasing degree of the lattice and ends at the greatest element of the lattice of degree k. The
    first derivative of an IPk path is minus the conditional mutual information. The critical dimension of an IP k path is the degree of its first minimum.
    A positive information path is an information path from 0 to a given I k corresponding to a given k-tuple of variables such that Ik<Ik-1<...<I1 .
    We call the interacting components functions Ik , k>1, a free information energy. A maximal positive information path is a positive information path
    of maximal length: it ends at minima of the free information energy function. The set of all these paths defines uniquely the minimum free energy complex.
    The set of all paths of degree k is intractable computationally (complexity in O(k!)). In order to bypass this issue, the algo computes a fast local
    algorithm that selects at each element of degree k of an IP path the positive information path with maximal or minimal Ik+1 value or stops whenever Xk.I k+1≤ 0
    and ranks those paths by their length.
    '''
    def information_complex( self, Ninfomut):

        infomut_per_order=[]
        Ninfomut_per_order_ordered=[]
        for x in range(self.dimension_max+1):
            info_dicoperoder={}
            infomut_per_order.append(info_dicoperoder)
            Ninfomut_per_order_ordered.append(info_dicoperoder)
        for x,y in Ninfomut.items():
            infomut_per_order[len(x)][x]=Ninfomut[x]
        for x in range(self.dimension_max+1):
            Ninfomut_per_order_ordered[x]=OrderedDict(sorted(infomut_per_order[x].items(), key=lambda t: t[1]))

        matrix_distrib_infomut=np.array([])
        x_absss = np.array([])
        y_absss = np.array([])
        maxima_tot=-1000000.00
        minima_tot=1000000.00
        infocond=1000000.00
        listartbis=[]
        infomutmax_path_VAR = []
        infomutmax_path_VALUE = []
        infomutmin_path_VAR = []
        infomutmin_path_VALUE = []
        number_of_max_and_min = self.dimension_max #explore the 2 max an min marginals (of degree 1 information)
        items = list(Ninfomut_per_order_ordered[1].items())
        for inforank in range(0,number_of_max_and_min):
            infomutmax_path_VAR.append([])
            infomutmax_path_VALUE.append([])
            infomutmin_path_VAR.append([])
            infomutmin_path_VALUE.append([])
            xstart=items[inforank][0]
            xstartmin = items[self.dimension_max - inforank - 1][0]   #items[self.dimension_max-inforank-1][0]
            infomutmax_path_VAR[-1].append(xstart[0])
            infomutmax_path_VALUE[-1].append(items[inforank][1])
            infomutmin_path_VAR[-1].append(xstartmin[0])
            infomutmin_path_VALUE[-1].append(items[self.dimension_max - inforank- 1][1])
            degree=1
            infocond=1000000.00
            while infocond >=0 and degree <= self.dimension_max:
                maxima_tot=-1000000.00
                minima_tot=1000000.00
                degree=degree+1
                for i in range(1,self.dimension_max+1) :
                    if i in infomutmax_path_VAR[-1] :
                        del listartbis[:]
                    else:
                        del listartbis[:]
                        listartbis=infomutmax_path_VAR[-1][:]
                        listartbis.append(i)
                        listartbis.sort()
                        tuplestart=tuple(listartbis)
                        if infomut_per_order[degree][tuplestart]>maxima_tot:
                            maxima_tot=infomut_per_order[degree][tuplestart]
                            igood= i
                infomutmax_path_VAR[-1].append(igood)
                infomutmax_path_VALUE[-1].append(maxima_tot)
                infocond= infomutmax_path_VALUE[-1][-2]- infomutmax_path_VALUE[-1][-1]
            del infomutmax_path_VAR[-1][-1]
            del infomutmax_path_VALUE[-1][-1]
            print('The path of maximal mutual-info Nb',inforank+1,' is :')
            print(infomutmax_path_VAR[-1])

            degree=1
            infocond=1000000.00
            while infocond >=0 :
                maxima_tot=-1000000.00
                minima_tot=1000000.00
                degree=degree+1
                for i in range(1,self.dimension_max+1) :
                    if i in infomutmin_path_VAR[-1] :
                        del listartbis[:]
                    else:
                        del listartbis[:]
                        listartbis=infomutmin_path_VAR[-1][:]
                        listartbis.append(i)
                        listartbis.sort()
                        tuplestart=tuple(listartbis)
                        if infomut_per_order[degree][tuplestart]<minima_tot:
                            minima_tot=infomut_per_order[degree][tuplestart]
                            igood= i
                infomutmin_path_VAR[-1].append(igood)
                infomutmin_path_VALUE[-1].append(minima_tot)
                infocond= infomutmin_path_VALUE[-1][-2]- infomutmin_path_VALUE[-1][-1]
            del infomutmin_path_VAR[-1][-1]
            del infomutmin_path_VALUE[-1][-1]
            print('The path of minimal mutual-info Nb',inforank+1,' is :')
            print(infomutmin_path_VAR[-1])

# COMPUTE THE HISTOGRAMS OF INFORMATION

        ListInfomutordre={}
        maxima_tot=-1000000.00
        minima_tot=1000000.00
        for i in range(1,self.dimension_max+1):
            ListInfomutordre[i]=[]
        for x,y in Ninfomut.items():
            ListInfomutordre[len(x)].append(y)
            if y>maxima_tot:
                maxima_tot=y
            if y<minima_tot:
                minima_tot=y

        for a in range(1,self.dimension_max+1):
            ListInfomutordre[a].append(minima_tot-0.1)
            ListInfomutordre[a].append(maxima_tot+0.1)
            n, bins = np.histogram(ListInfomutordre[a],  bins = self.nb_bins_histo)
            if a==1 :
                matrix_distrib_infomut=n
            else:
                matrix_distrib_infomut=np.c_[matrix_distrib_infomut,n]

# COMPUTE THE MATRIX OF INFORMATION LANDSACPES

        num_fig=1
        fig_infopath = plt.figure(num_fig,figsize=(18, 10))
        matrix_distrib_infomut=np.flipud(matrix_distrib_infomut)
        plt.matshow(matrix_distrib_infomut, cmap='jet', aspect='auto', extent=[0,self.dimension_max,minima_tot-0.1,maxima_tot+0.1], norm=LogNorm(vmin=1, vmax=200000), fignum= num_fig)
        plt.axis([0,self.dimension_max,minima_tot,maxima_tot])
        cbar = plt.colorbar()
        cbar.set_label('# of tuples', rotation=270)
        plt.grid(False)


# COMPUTE THE INFORMATION PATHS
        x_infomax=[]
        x_infomin=[]
        maxima_x=-10
        maxima_tot=-1000000.00
        minima_tot=1000000.00
        for inforank in range(0,number_of_max_and_min):
            x_infomax.append([])
            j=-0.5
            for y in  range(0,len(infomutmax_path_VALUE[inforank])):
                j=j+1
                x_infomax[-1].append(j)
                if j > maxima_x:
                    maxima_x=j
                if infomutmax_path_VALUE[inforank][int(j-0.5)]>maxima_tot :
                    maxima_tot=infomutmax_path_VALUE[inforank][int(j-0.5)]
                if infomutmax_path_VALUE[inforank][int(j-0.5)]<minima_tot :
                    minima_tot=infomutmax_path_VALUE[inforank][int(j-0.5)]

            x_infomin.append([])
            j=-0.5
            for y in  range(0,len(infomutmin_path_VALUE[inforank])):
                j=j+1
                x_infomin[-1].append(j)
                if j > maxima_x:
                    maxima_x=j
                if infomutmin_path_VALUE[inforank][int(j-0.5)]>maxima_tot :
                    maxima_tot=infomutmin_path_VALUE[inforank][int(j-0.5)]
                if infomutmin_path_VALUE[inforank][int(j-0.5)]<minima_tot :
                    minima_tot=infomutmin_path_VALUE[inforank][int(j-0.5)]

            plt.plot(x_infomax[inforank], infomutmax_path_VALUE[inforank], marker='o', color='red')
            plt.plot(x_infomin[inforank], infomutmin_path_VALUE[inforank], marker='o',color='blue')
            plt.axis([0,maxima_x+0.5,minima_tot-0.2,maxima_tot+0.2])


        display_labelnodes=False
        if display_labelnodes :
            for inforank in range(0,number_of_max_and_min):
                for label, x,y in zip(infomutmax_path_VAR[inforank], x_infomax[inforank], infomutmax_path_VALUE[inforank]):
                    plt.annotate(label,xy=(x, y), xytext=(0, 0),textcoords='offset points')
                for label, x,y in zip(infomutmin_path_VAR[inforank], x_infomin[inforank], infomutmin_path_VALUE[inforank]):
                    plt.annotate(label,xy=(x, y), xytext=(0, 0),textcoords='offset points')
        plt.title('Ik paths - information - Free Energy complex')
        plt.xlabel('dimension')
        plt.ylabel('Ik value (bits)')
        fig_infopath.set_size_inches(18, 10)
        plt.grid(False)
        plt.show()
