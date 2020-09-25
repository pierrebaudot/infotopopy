"""
Example using the iris dataset
==============================

This example illustrates infotopo using the Iris dataset.
"""
from infotopo import Infotopo
from infotopo.io import load_data_sets

import matplotlib.pyplot as plt


###############################################################################
# Loading the dataset
###############################################################################
# First, start by loading the data and make the plot of the distributions

dataset, nb_of_values = load_data_sets(1, plot_data=True)
plt.show()

###############################################################################
# Define an infotopo object
###############################################################################
# Then we defined an infotopo example

dimension_max = dataset.shape[1]
dimension_tot = dataset.shape[1]
sample_size = dataset.shape[0]
forward_computation_mode = False
work_on_transpose = False
supervised_mode = False
sampling_mode = 1
deformed_probability_mode = False   

information_topo = Infotopo(
    dimension_max=dimension_max,
    dimension_tot=dimension_tot,
    sample_size=sample_size,
    work_on_transpose=work_on_transpose,
    nb_of_values=nb_of_values,
    sampling_mode=sampling_mode,
    deformed_probability_mode=deformed_probability_mode,
    supervised_mode=supervised_mode,
    forward_computation_mode=forward_computation_mode)

Nentropie = information_topo.simplicial_entropies_decomposition(dataset) 
information_topo.entropy_simplicial_lanscape(Nentropie)