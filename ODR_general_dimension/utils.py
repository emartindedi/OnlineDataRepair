import math
import random
import pandas as pd
import numpy as np

'''
    File name: utils.py
    Description: Scripts of function used in another classes
    Author: Elena M. De Diego
    GitHub: https://github.com/emartindedi/OnlineDataRepair.git
    Package: OnlineDataRepair
    Python Version: 3.8 and 3.9
    License: No use withut explicit consention
'''


def simulated_data_general_dimension(n_0, n_1, dim):
    """
    Function to generate a sintetic dataset according to a normal distribution
    :param n_0: integer with the number of samples of the minoritary group
    :param n_1: integer with the number of samples of the default group
    :dim: dimension (number of features of the expected dataset)
    :return: pandas dataframe object
    """
    np.random.seed(1999)

    mu_0, mu_1, sig, columns = [], [], [], []
    for d in range(dim):
        mu_0.append(round(random.uniform(1, 4), 1))
        mu_1.append(round(random.uniform(1.5, 5.5), 1))
        sig.append(round(random.uniform(0, 1), 1))
        columns.append('col_' + str(d))

    mu_0 = np.array(mu_0)
    mu_1 = np.array(mu_1)
    sigma = np.diag(sig)

    minority = np.random.multivariate_normal(mu_0, sigma, size=n_0)  # numpy array
    default = np.random.multivariate_normal(mu_1, sigma, size=n_1)  # numpy array

    X = np.concatenate((minority, default), axis=0)

    var_protected = np.array([1 if i > n_0 - 1 else 0 for i in range(n_0 + n_1)])

    data = pd.DataFrame(data=X, columns=columns)
    data["protected"] = var_protected

    return data

def simulated_data_general_dimension_plus_new(n_0, n_1, n_0_new, n_1_new, dim):
    """
    Function to generate a sintetic dataset according to a normal distribution
    :param n_0: integer with the number of samples of the minoritary group
    :param n_1: integer with the number of samples of the default group
    :dim: dimension (number of features of the expected dataset)
    :return: pandas dataframe object
    """
    np.random.seed(1999)

    mu_0, mu_1, sig, columns = [], [], [], []
    for d in range(dim):
        mu_0.append(round(random.uniform(1, 4), 1))
        mu_1.append(round(random.uniform(2, 7), 1))
        sig.append(round(random.uniform(0, 1), 1))
        columns.append('col_' + str(d))

    mu_0 = np.array(mu_0)
    mu_1 = np.array(mu_1)
    sigma = np.diag(sig)

    """Original"""
    minority = np.random.multivariate_normal(mu_0, sigma, size=n_0)  # numpy array
    default = np.random.multivariate_normal(mu_1, sigma, size=n_1)  # numpy array

    X = np.concatenate((minority, default), axis=0)

    var_protected = np.array([1 if i > n_0 - 1 else 0 for i in range(n_0 + n_1)])

    data_original = pd.DataFrame(data=X, columns=columns)
    data_original["protected"] = var_protected

    """New"""
    minority_new = np.random.multivariate_normal(mu_0, sigma, size=n_0_new)  # numpy array
    default_new = np.random.multivariate_normal(mu_1, sigma, size=n_1_new)  # numpy array

    X_new = np.concatenate((minority_new, default_new), axis=0)

    var_protected_new = np.array([1 if i > n_0_new - 1 else 0 for i in range(n_0_new + n_1_new)])

    data_new = pd.DataFrame(data=X_new, columns=columns)
    data_new["protected"] = var_protected_new

    return (data_original, data_new)

def cyclically_monotone_general_dimension(matrix_X, matrix_Y):

    """
    Function to check if a set is cyclically monotone.
    The set is {(x,y): x in vector_X y in vector_Y}
    :param vector_X pandas Dataframe
    :param vector_Y pandas Dataframe
    :return: boolean [True/False]
    """

    n = matrix_X.shape[0] # tamaño del conjunto S en el papar 'Distributions and Quantile'
    cyclically_monotone_bool = True # answer to the question Is the set {(x,y): x in vector_X y in vector_Y} cyclically monotone?
    non_stop = True
    #matrix_X_mod = np.zeros(shape=matrix_X.shape[1])

    while cyclically_monotone_bool and non_stop:

        for k in range(n):
            "For any finite collection of points..."
            sum = 0
            # x_{k+1} = x_{1}
            if k != n-1:
                matrix_X_k_plus_1 = matrix_X.iloc[k+1,:]
            else:
                matrix_X_k_plus_1 = matrix_X.iloc[0,:]

            for i in range(k+1):
                sum += np.inner(matrix_Y.iloc[i,:], matrix_X_k_plus_1-matrix_X.iloc[i,:])

            if sum <= 0:
                cyclically_monotone_bool = True
            else:
                cyclically_monotone_bool = False

        if k == n-1:
            non_stop = False # Stop the while loop

    print('Is the set {(x,y): x in matrix_X y in matrix_Y} cyclically monotone?: ', cyclically_monotone_bool)

    return cyclically_monotone_bool