"""Load dataset and examples.

This function loads different data set for unitary test or tutorial purpose,
either theoretical-synthetic or real dataset from scikit-learn
https://scikit-learn.org/stable/datasets/index.html  
    * dataset = 1 load iris dataset
    * dataset = 2 load Boston house prices
    * dataset = 3 load DIABETES  dataset
    * dataset = 4 CAUSAL Inference data challenge
      http://www.causality.inf.ethz.ch/data/LUCAS.html
    * dataset = 5 Borromean
    * dataset = 6 mnist digit dataset
"""
import matplotlib.pyplot as plt



def load_data_sets(dataset_type, plot_data=False):
    """Loading example dataset.

    Parameters
    ----------
    dataset_type : int
        Dataset type to load. Use either :

            * 1 : iris dataset
            * 2 : boston dataset
            * 3 : diabete dataset
            * 4 : causal inference data challenge
            * 5 : Borromean case I_1 are 1 bit (max: "random")  I_2 are 0 bit
              (min: independent) I_3 is -1 bit
            * 6 : mnist digit dataset

    plot_data : bool | False
        Plot the data (True)

    Returns
    -------
    dataset : array_like
        Array of data
    nb_of_values : int
        Number of values
    """
    import pandas as pd
    import seaborn as sns
    from sklearn.datasets import load_iris, load_digits, load_boston, load_diabetes

    if dataset_type == 1:  # Iris
        dataset = load_iris()
        dataset_df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        nb_of_values = 9
        dataset_df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        dataset_df['species'] = pd.Series(dataset.target).map(dict(zip(range(3),dataset.target_names)))
        if plot_data:
            sns.pairplot(dataset_df, hue='species')
        dataset = dataset.data
    elif dataset_type == 2:  # Boston
        dataset = load_boston()
        dataset_df = pd.DataFrame(dataset.data, columns = dataset.feature_names)
        nb_of_values =9
        dataset_df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        dataset_df['MEDV'] = pd.Series(dataset.target).map(dict(zip(range(3),dataset.data[:,12])))
        dataset = dataset.data
    elif dataset_type == 3:  # diabetes
        dataset = load_diabetes()
        dataset_df = pd.DataFrame(dataset.data, columns = dataset.feature_names)
        nb_of_values = 9
        dataset_df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        dataset = dataset.data
    elif dataset_type == 4:  # causal inference
        dataset = pd.read_csv(r"/home/pierre/Documents/Data/lucas0_train.csv")  # csv to download at http://www.causality.inf.ethz.ch/data/LUCAS.html
        print(dataset.columns)
        print(dataset.shape)
        dataset_df = pd.DataFrame(dataset, columns = dataset.columns)
        dataset = dataset.to_numpy()
        nb_of_values = 2           
    elif dataset_type == 5:  # borromean case
        nb_of_values = 2
        if nb_of_values == 2:
                dataset = np.array([[ 0,  0,  1],
                                    [ 0,  1,  0],
                                    [ 1,  0,  0],
                                    [ 1,  1,  1]])
        elif nb_of_values == 3:
                dataset = np.array([[ 0,  0,  0],
                                    [ 1,  1,  0],
                                    [ 2,  2,  0],
                                    [ 0,  1,  1],
                                    [ 1,  2,  1],
                                    [ 2,  0,  1],
                                    [ 0,  2,  2],
                                    [ 1,  0,  2],
                                    [ 2,  1,  2]])  
        elif nb_of_values == 4:
                dataset = np.array([[ 3,  0,  0],
                                    [ 2,  1,  0],
                                    [ 1,  2,  0],
                                    [ 0,  3,  0],
                                    [ 0,  0,  1],
                                    [ 1,  3,  1],
                                    [ 2,  2,  1],
                                    [ 3,  1,  1],
                                    [ 1,  0,  2],
                                    [ 0,  1,  2],
                                    [ 2,  3,  2],
                                    [ 3,  2,  2],
                                    [ 0,  2,  3],
                                    [ 1,  1,  3],
                                    [ 2,  0,  3],
                                    [ 3,  3,  3]])    
    elif dataset_type == 6:  # mnist digit dataset
        dataset = load_digits()
        print(dataset.DESCR)
        fig, ax_array = plt.subplots(20, 20)
        axes = ax_array.flatten()
        for i, ax in enumerate(axes):
            ax.imshow(dataset.images[i], cmap='gray_r')
        plt.setp(axes, xticks=[], yticks=[], frame_on=False)
        plt.tight_layout(h_pad=0.5, w_pad=0.01)
        nb_of_values = 17
        dataset = dataset.data    
        plt.show()                                                                                           
    return dataset, nb_of_values 
