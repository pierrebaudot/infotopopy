import pandas as pd
import seaborn as sns

import timeit

from infotopo import Infotopo
from infotopo.io import load_data_sets


dataset_type = 1 # if dataset = 1 load IRIS DATASET # if dataset = 2 load Boston house prices dataset # if dataset = 3 load DIABETES  dataset 
## if dataset = 4 CAUSAL Inference data challenge http://www.causality.inf.ethz.ch/data/LUCAS.html  # if dataset = 5 Borromean  dataset
# if dataset = 6 Digits dataset MNIST
dataset, nb_of_values = load_data_sets( dataset_type)
dimension_max = dataset.shape[1]
dimension_tot = dataset.shape[1]
sample_size = dataset.shape[0]
forward_computation_mode = False
work_on_transpose = False
supervised_mode = False
sampling_mode = 1
deformed_probability_mode = False     
if dataset_type == 6:
    forward_computation_mode = True
    dimension_max = 5

print("sample_size : ",sample_size)
print('number of variables or dimension of the analysis:',dimension_max )
print('number of tot  dimensions:',  dimension_tot)
print('number of values:', nb_of_values)

information_topo = Infotopo(dimension_max = dimension_max, 
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
Nentropie = information_topo.simplicial_entropies_decomposition(dataset) 
stop = timeit.default_timer()
print('Time for CPU(seconds) entropies: ', stop - start)
if dataset_type == 1 or dataset_type == 5:
    print(Nentropie)
information_topo.entropy_simplicial_lanscape(Nentropie)
information_topo = Infotopo(dimension_max = dimension_max, 
                            dimension_tot = dimension_tot, 
                            sample_size = sample_size, 
                            work_on_transpose = work_on_transpose,
                            nb_of_values = nb_of_values, 
                            sampling_mode = sampling_mode, 
                            deformed_probability_mode = deformed_probability_mode,
                            supervised_mode = supervised_mode, 
                            forward_computation_mode = forward_computation_mode,
                            dim_to_rank = 3, number_of_max_val = 4)
if dataset_type != 5:
    dico_max, dico_min = information_topo.display_higher_lower_information(Nentropie, dataset)

# Ninfomut is a dictionary (x,y) with x a list of kind (1,2,5) and y a value in bit
Ntotal_correlation = information_topo.total_correlation_simplicial_lanscape(Nentropie)
dico_max, dico_min = information_topo.display_higher_lower_information(Ntotal_correlation, dataset)
start = timeit.default_timer()   
Ninfomut = information_topo.simplicial_infomut_decomposition(Nentropie)
stop = timeit.default_timer()
print('Time for CPU(seconds) Mutual Information: ', stop - start)
if dataset_type == 1 or dataset_type == 5:
    print(Ninfomut)
information_topo.mutual_info_simplicial_lanscape(Ninfomut)   
if dataset_type != 5: 
    dico_max, dico_min = information_topo.display_higher_lower_information(Ninfomut, dataset)
adjacency_matrix_mut_info = information_topo.mutual_info_pairwise_network(Ninfomut)
mean_info, mean_info_rate  =information_topo.display_mean_information(Ninfomut)
# CONDITIONAL INFO OR ENTROPY
NcondInfo = information_topo.conditional_info_simplicial_lanscape(Ninfomut)
information_topo.display_higher_lower_cond_information(NcondInfo)
# ENTROPY vs. ENERGY LANDSCAPE
information_topo.display_entropy_energy_landscape(Ntotal_correlation, Nentropie)
information_topo.display_entropy_energy_landscape(Ninfomut, Nentropie)
# Information distance and volume LANDSCAPE
Ninfo_volume = information_topo.information_volume_simplicial_lanscape(Nentropie, Ninfomut)
dico_max, dico_min = information_topo.display_higher_lower_information(Ninfo_volume, dataset)
adjacency_matrix_info_distance = information_topo.mutual_info_pairwise_network(Ninfo_volume)
# Information paths - Information complex
Ninfomut, Nentropie =  information_topo.fit(dataset)
information_topo.information_complex(Ninfomut)