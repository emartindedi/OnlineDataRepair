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


def simulated_data(n_0, n_1):
    """
    Function to generate a sintetic dataset according to a normal distribution
    :param n_0: integer with the number of samples of the minoritary group
    :param n_1: integer with the number of samples of the default group
    :return: pandas dataframe object
    """
    np.random.seed(1999)
    mu_0, sigma_0 = 5, 2 # Media and standard deviation for minority group
    mu_1, sigma_1 = 8, math.sqrt(1) # Media and standard deviation for default group

    minority = np.random.normal(mu_0, sigma_0, n_0)  # numpy array
    default = np.random.normal(mu_1, sigma_1, n_1)  # numpy array

    X = np.concatenate((minority, default), axis=0)

    var_protected = np.array([1 if i > n_0 - 1 else 0 for i in range(n_0 + n_1)])

    data = pd.DataFrame(data=X, columns=["education"])
    data["sex"] = var_protected

    return data

def divide_by_protected(df, name_column, type_protected, minority, default):
    """
    Function that divide the original dataset into two different datasets according to the value of the
    protected variable

    :df: pandas dataframe,
    :name_column: string with the name of the protected variable
    :minority: value of the minority class in the name_column variable
    :default: value of the minority class in the name_column variable
    :return: tuple with two different dataframes
    """
    if type_protected == 'Integer':

        df_min = df[df[name_column] == int(minority)]
        df_def = df[df[name_column] == int(default)]

        dataset_minority = df_min
        dataset_default = df_def

    elif type_protected == 'String':

        df_min = df[df[name_column] == minority]
        df_def = df[df[name_column] == default]
        dataset_minority = df_min
        dataset_default = df_def

    else:

        try:
            df_min = df[df[name_column] == minority]
            df_def = df[df[name_column] == default]

            dataset_minority = df_min
            dataset_default = df_def

        except Exception as e:
            print(e)

    return (df_min, df_def)

def extract_column(df, name_column):
    """
    Function that extract a column selected of a dataset.

    :df: pandas dataframe with the name of the columns included
    :name_column: string with the name of the column that will be returned
    :return: pandas series with the column of df selected
    """
    return df[name_column]

def cyclically_monotone(vector_X, vector_Y):
    """
    Function to check if a set is cyclically monotone.
    The set is {(x,y): x in vector_X y in vector_Y}
    :param vector_X numpy array
    :param vector_Y numpy array
    :return: boolean [True/False]
    """
    n = len(vector_X) # tamaño del conjunto S en el papar 'Distributions and Quantile'
    cyclically_monotone_bool = True # answer to the question Is the set {(x,y): x in vector_X y in vector_Y} cyclically monotone?
    non_stop = True
    while cyclically_monotone_bool and non_stop:
        for k in range(n):
            "For any finite collection of points..."
            sum = 0
            # x_{k+1} = x_{1}
            vector_X_mod = vector_X[:k+1]
            vector_X_mod = np.append(vector_X_mod, vector_X[0])
            for i in range(k+1):
                sum += np.inner(vector_Y[i], vector_X_mod[i+1]-vector_X[i])
            if sum <= 0:
                cyclically_monotone_bool = True
            else:
                cyclically_monotone_bool = False
        if k == n-1:
            non_stop = False # Stop the while loop
    print('Is the set {(x,y): x in vector_X y in vector_Y} cyclically monotone?: ', cyclically_monotone_bool)
    return cyclically_monotone_bool