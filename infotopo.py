#!/usr/bin/env python
# coding: utf-8


# list of dependencies
#import 
import math
import numpy as np
import itertools
import timeit
from itertools import combinations, chain
import logging
from decimal import Decimal
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import networkx as nx


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

class infotopo:
    """
    infotopo : 
    computes the simplicial information cohomology of a set of variable, notably the joint and conditional entropies, 
    the mutual and conditional mutual information, total correlations, and information paths within the simplicial set 

    Parameters:
    dimension_max : (integer) maximum Nb  of Random Variable (column or dimension) for the exploration of the cohomology and lattice 

    dimension_tot : (integer) total Nb of Random Variable (column or dimension)  to consider in the input matrix for analysis (the first columns)

    sample_size : (integer) total Nb of points (rows or number of trials)  to consider in the input matrix for analysis (the first rows)

    work_on_transpose :(Boolean) if True take the transpose of the input matrix (this change column into rows etc.)

    nb_of_values : (integer) Number of different values for the sampling of each variable (alphabet size)

    sampling_mode : (integer: 1,2,3) 
                        _ sampling_mode = 1: normalization taking the max and min of each columns (normaization row by columns)
                        _ sampling_mode = 2: normalization taking the max and min of the whole matrix
    
    deformed_probability_mode: (Boolean) 
                        _ deformed_probability_mode = True : it will compute the "escort distribution" also called the "deformed probabilities".
                        p(n,k)= p(k)^n/ (sum(i)p(i)^n   , where n is the sample size. 
                        [1] Umarov, S., Tsallis C. and Steinberg S., On a q-Central Limit Theorem Consistent with Nonextensive Statistical Mechanics, Milan j. math. 76 (2008), 307–328
                        [2] Bercher,  Escort entropies and divergences and related canonical distribution. Physics Letters A Volume 375, Issue 33, 1 August 2011, Pages 2969-2973
                        [3] A. Chhabra, R. V. Jensen, Direct determination of the f(α) singularity spectrum.  Phys. Rev. Lett. 62 (1989) 1327.
                        [4] C. Beck, F. Schloegl, Thermodynamics of Chaotic Systems, Cambridge University Press, 1993.
                        [5] Zhang, Z., Generalized Mutual Information.  July 11, 2019
                        _ deformed_probability_mode = False : it will compute the classical probability, e.g. the ratio of empirical frequencies over total number of observation
                        [6] Kolmogorov 1933 foundations of probability                     

    supervised_mode : (Boolean) if True it will consider the lavelvector for supervised learning; if False unsupervised mode

    forward_computation_mode: (Boolean) 
                        _ forward_computation_mode = True : it will compute joint entropies on the simplicial lattice from low dimension 
                        to high dimension (co-homological way). For each element of the lattice of random-variable the corresponding joint 
                        probability is estimated. This allows to explore only the first low dimensions-rank of the lattice, up to dimension_max
                        (in dimension_tot)
                        _ forward_computation_mode = False : it will compute joint entropies on whole  simplicial lattice from high dimension 
                        to the marginals (homological way). The joint probability corresponding to all variable is first estimated and then projected on 
                        lower dimensions using conditional rule. This explore the whole lattice, and imposes dimension_max = dimension_tot   

    nb_bins_histo : (integer) number of values used for entropy and mutual information distribution histograms and lsample_sizeandscapes.          

    compute_shuffle : (Boolean)
                        _ compute_shuffle = True : it will compute the statictical test of significance of the dependencies (pethel et hah 2014) 
                        and make shuffles that preserve the marginal but the destroys the mutual informations 
                        _  compute_shuffle = False : no shuffles and test of the mutual information estimations is acheived

    p_value :       (real in ]0,1[) p value of the test of significance of the dependencies estimated by mutual info 
                    the H0 hypotheis is the mutual Info distribution does not differ from the distribution of MI with shuffled higher order dependencies
    
    nb_of_shuffle: (integer) number of shuffles computed                              

    """
    def __init__(self, 
        dimension_max = 16, 
        dimension_tot = 16, 
        sample_size = 1000, 
        work_on_transpose = False,
        nb_of_values = 9, 
        sampling_mode = 1, 
        deformed_probability_mode = False,
        supervised_mode = False, 
        forward_computation_mode = False,
        nb_bins_histo = 200,
        compute_shuffle = False,
        p_value = 0.05, 
        nb_of_shuffle = 20):

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
        self.compute_shuffle = compute_shuffle
        self.p_value = p_value
        self.nb_of_shuffle = nb_of_shuffle

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
        if not self.compute_shuffle:
            self.nb_of_shuffle = 0
                


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
    """                

    def _resample_matrix(self, data_matrix):
        if self.work_on_transpose: 
            data_matrix.transpose()
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
        plt.figure(num_fig)
        moyenne={}
        nbpoint={}
        matrix_distrib_info=np.array([])
        x_absss = np.array([])
        y_absss = np.array([])
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
                if ListEntropyordre[a][x] >= ((math.log(self.sample_size)/math.log(2))-0.00000001):
                    nb_undersampling_point=nb_undersampling_point+1
                    if a ==1:
                        print(ListEntropyordre[a][x])
            undersampling_percent = np.hstack((undersampling_percent,100*nb_undersampling_point/self._binomial(self.dimension_max,a)))
            print('undersampling_percent in dim ',a,' = ', 100*nb_undersampling_point/self._binomial(self.dimension_max,a))
            ListEntropyordre[a].append(minima_tot-0.1)
            ListEntropyordre[a].append(maxima_tot+0.1)
            n, bins, patches = plt.hist(ListEntropyordre[a], self.nb_bins_histo, facecolor='g')
            plt.axis([minima_tot, maxima_tot,0,n.max()])
            if a==1 :
                matrix_distrib_info=n
            else:
                matrix_distrib_info=np.c_[matrix_distrib_info,n]
            plt.grid(True)
    
        num_fig=num_fig+1
        plt.figure(num_fig)
        abssice_degree=np.linspace(1, self.dimension_max, self.dimension_max)
        plt.plot(abssice_degree,undersampling_percent)
        plt.ylabel('(percent of undersampled points)')
        plt.title('undersampling bound')
        plt.grid(True)

        num_fig=num_fig+1
        plt.figure(num_fig)
        matrix_distrib_info=np.flipud(matrix_distrib_info)
        plt.matshow(matrix_distrib_info, cmap='jet', aspect='auto', extent=[0,self.dimension_max,minima_tot-0.1,maxima_tot+0.1], norm=LogNorm())
        plt.axis([0,self.dimension_max,minima_tot,maxima_tot])
        plt.colorbar()
        plt.grid(False)
        plt.show(num_fig)

#########################################################################
#########################################################################
######      Histogramms MUTUAL INFORMATION      #########################
######           &  INFOMUT LANDSCAPES          #########################
#########################################################################
#########################################################################
    def mutual_info_simplicial_lanscape(self, Ninfomut) :
        num_fig = 1
        plt.figure(num_fig)
        matrix_distrib_infomut=np.array([])
        x_absss = np.array([])
        y_absss = np.array([])
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
#        Ninfomut.clear()
#        del(Ninfomut)
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
                Ninfomut.clear()
                del(Ninfomut)
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
#   COMPUTE THE HISTOGRAMS OF THE LIST OF INFOMUT VALUES FOR EACH DEGREE
#   If shuffle is true it also compute the signifiance test against independence null hypothesis
        num_fig=num_fig+1
        fig_Histo_infomut = plt.figure(num_fig)
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
            if a==1 :
                matrix_distrib_infomut=n
            else:
                matrix_distrib_infomut=np.c_[matrix_distrib_infomut,n]
            plt.grid(True)
      
#   COMPUTE THE INFOMUT LANDSCAPE FROM THE HISTOGRAMS AS THE MATRIX  matrix_distrib_infomut
#   If shuffle is true it also plots the signifiance test against independence null hypothesis
        num_fig=num_fig+1
        fig_infolandscape =plt.figure(num_fig)
        matrix_distrib_infomut=np.flipud(matrix_distrib_infomut)
        plt.matshow(matrix_distrib_infomut, cmap='jet', aspect='auto', extent=[0,self.dimension_max,minima_tot-0.1,maxima_tot+0.1], norm=LogNorm())
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
        plt.colorbar()
        plt.grid(False)
        plt.show(num_fig)
     
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
        print(infomut_per_order)
        num_fig = 1
        plt.figure(num_fig)
        netring = nx.Graph()
        list_of_node=[]
        list_of_size=[]
        list_of_edge=[]
        list_of_width=[]
        list_of_labels={}
        for x in range(1,self.dimension_max+1):
            tuple_interim=(x)
            list_of_node.append(x)
            list_of_labels[x]=tuple_interim
            number_node=[x]
            code_var=tuple(number_node)
            print("infomut_per_order[1][(x,)]",infomut_per_order[1][(x,)])
            list_of_size.append((infomut_per_order[1][(x,)]**2) *500)
            netring.add_node(x)
        tuple_interim=()
        for x,y in infomut_per_order[2].items():
#            tuple_interim=(workbook_data[Name_worksheet].cell(row = x[0]+1, column = 1).value,workbook_data[Name_worksheet].cell(row = x[1]+1, column = 1).value)
            list_of_edge.append(x)
            var_1=x[0]
            var_2=x[1]
            netring.add_edge(var_1, var_2, weight= (infomut_per_order[2][x]) )
            list_of_width.append((infomut_per_order[2][x]*10))
        print(list_of_node)
        nx.draw_circular(netring, with_labels= True, nodelist = list_of_node,edgelist = list_of_edge, width= list_of_width, node_size = list_of_size)
        plt.show()


# #########################################################################
# #########################################################################
# ######          MAIN PROGRAM               ##############################
# #########################################################################
# #########################################################################

if __name__ == "__main__":
    from sklearn.datasets import load_iris, load_digits
    import pandas as pd
    import seaborn as sns
    
    iris = load_iris()
    iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
    dimension_max = iris.data.shape[1]
    dimension_tot = iris.data.shape[1]
    sample_size = iris.data.shape[0]
    nb_of_values =9
    forward_computation_mode = False
    work_on_transpose = False
    supervised_mode = False
    sampling_mode = 1
    deformed_probability_mode = False

    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['species'] = pd.Series(iris.target).map(dict(zip(range(3),iris.target_names)))
    sns.pairplot(iris_df, hue='species')
    plt.show()

    
    print("sample_size : ",iris.data.shape[0])
    print('number of variables:',iris.data.shape[1])
    print('number of tot variables dimensions:', iris.data.shape[1])
    print('number of values:', nb_of_values)
    information_topo = infotopo(dimension_max = dimension_max, 
                                dimension_tot = dimension_tot, 
                                sample_size = sample_size, 
                                work_on_transpose = work_on_transpose,
                                nb_of_values = nb_of_values, 
                                sampling_mode = sampling_mode, 
                                deformed_probability_mode = deformed_probability_mode,
                                supervised_mode = supervised_mode, 
                                forward_computation_mode = forward_computation_mode)
# Nentropy is dictionary (x,y) with x a list of kind (1,2,5) and y a value in bit    
    start = timeit.default_timer()
    Nentropie = information_topo.simplicial_entropies_decomposition(iris.data)
    stop = timeit.default_timer()
    print('Time for CPU(seconds) entropies: ', stop - start)
    print(Nentropie)
    information_topo.entropy_simplicial_lanscape(Nentropie)

# Ninfomut is dictionary (x,y) with x a list of kind (1,2,5) and y a value in bit
    start = timeit.default_timer()   
    Ninfomut = information_topo.simplicial_infomut_decomposition(Nentropie)
    stop = timeit.default_timer()
    print('Time for CPU(seconds) Mutual Information: ', stop - start)
    print(Ninfomut)
    information_topo.mutual_info_simplicial_lanscape(Ninfomut)



    information_topo.mutual_info_pairwise_network(Ninfomut)

 #  key for key in Ninfomut if len(key)==2

        


