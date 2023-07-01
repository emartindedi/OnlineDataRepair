import numpy as np
import pandas as pd

'''
    File name: karps_algorithm.py
    Description: Script to solve the linear program 
        (equation 3.6 right after the corollary 3.2) of https://arxiv.org/pdf/1806.01238
    Author: Elena M. De Diego
    GitHub: https://github.com/emartindedi/OnlineDataRepair.git
    Package: OnlineDataRepair
    Python Version: 3.8 and 3.9
    License: No use withut explicit consention
'''

class Karp:

    r"""
    Implementation of the Karp's algorithm (Karp 1978)

    Class that implement the Karp's algorithm (Karp 1978) applied to the
    problem described in Hallin, Marc & del Barrio, Eustasio & Cuesta-Albertos, Juan & Matrán, Carlos. (2021).
    Distribution and quantile functions, ranks and signs in dimension d: A measure transportation approach. The Annals of Statistics. 49. 10.1214/20-AOS1996
    <https://www.researchgate.net/publication/350593971_Distribution_and_quantile_functions_ranks_and_signs_in_dimension_d_A_measure_transportation_approach>`_.

    Notes
    -----
    The linear program to solve is the equation 3-6 of the previous paper.
    The solution is computed in O(n^{3}) computer time

    """

    psi : None
    epsilon_star : float
    solution_karp : None


    def __init__(self, matrix_X, matrix_Y):
        """Initialize a new instance.

        Parameters
        ----------
        matrix_X : n tuple of points in R^{d}. (pandas Dataframe)
        vector_Y : n tuple of points in R^{d}. (pandas Dataframe)

        Notes
        ------
        Both of them needs to be an instance of pandas Dataframe
        """

        if not (isinstance(matrix_X, pd.DataFrame) or isinstance(matrix_X, pd.Series)):
            raise TypeError('The matrix_X is not a required pandas dataframe object or pandas series object')
        if not (isinstance(matrix_Y, pd.DataFrame) or isinstance(matrix_Y, pd.Series)):
            raise TypeError('The matrix_Y is not a required pandas dataframe object or pandas series object')

        self.matrix_X = matrix_X
        self.matrix_Y = matrix_Y

    def calculate_c_i_j(self, n, vector_X, vector_Y):
        """
        Function to calculate c_{i,j}  := <x_{i}, y_{i}-y_{j}>

        Parameters
        ----------
        n : length of the vector X
        vector_X, vector_Y : the n-tuples of points in R^{d}
        """
        c = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                c[i, j] = np.inner(vector_X.iloc[i,:], vector_Y.iloc[i,:] - vector_Y.iloc[j,:])
        return c

    def calculate_d_k_i(self, n, c):
        """
        Function to calculate d_{k,i} o<=k<=n 0<=i<=n

        Parameters
        ----------
        n : number of rows of the matrix X
        c : the output of the function calculate_c_i_j (matrix)
        """

        d = np.zeros((n + 1, n))
        cc = np.copy(c)  # a copy to return to the values in c original

        # c_{i,i} = inf
        for i in range(n):
            cc[i, i] = 1000000  # c[i,i] = float('inf')

        # d_{0,0} = 0; d_{0,i} = inf for i !=0
        for i in range(n):
            if i != 0:
                d[0, i] = 1000000  # d[0,i] = float('inf') # quasi inf

        # calculate d_{k,i} recursively
        for k in range(1, n + 1):
            for i in range(n):
                # print('(k,i)', k,  i)
                if k == 1:
                    d[1, i] = cc[0, i]
                else:
                    valores_a_comparar = np.array([0.0 for l in range(n)])  # create, it does not matter the zero values
                    for j in range(n):
                        valores_a_comparar[j] = d[k - 1, j] + cc[j, i]
                    # print(valores_a_comparar)
                    d[k, i] = np.amin(valores_a_comparar)
                # print(f"d_{k}_{i}".format(k,i), ": ", d[k,i])
        return d

    def calculate_epsilon_star(self, n, d):
        """
        Function to calculate e* and returns also if epsilon_star is strictly greater than zero which is equivalent to
        the assumption A

        Parameters
        ----------
        n : number of rows of the matrix X
        d : output of the function calculate_d_k_i
        :return: ([True/False], epsilon_star)"""

        epsilon_star = 0.0
        maximo = np.array([0.0 for l in range(n)])

        for i in range(n):
            # print('i', i)
            comparacion_maximo = np.array([0.0 for l in range(n)])
            for k in range(n):
                # print('k', k)
                comparacion_maximo[k] = (d[n, i] - d[k, i]) / (n - k)
            # print(comparacion_maximo)
            maximo[i] = np.amax(comparacion_maximo)

        epsilon_star = np.amin(maximo)
        if epsilon_star > 0:
            asumption_A = True
            print("Assumption A: ", True)
        else:
            asumption_A = False
            print("Assumption A: ", False)
        return (asumption_A, epsilon_star)

    def calculate_c_i_j_tilda(self, c, epsilon_star):
        """
        Function to calculate the c_i_j_tilda; Graph with modified costs c_i_j_tilda = c_i_j - epsilon_star

        Parameters
        ----------
        c : c matrix
        epsilon_star: float value previously calculated
        """
        c_tilda = np.zeros(c.shape)
        for i in range(self.matrix_X.shape[0]):
            for j in range(self.matrix_X.shape[0]):
                c_tilda[i, j] = c[i, j] - epsilon_star
        return c_tilda

    def calculate_d_i_tilda(self, n, d_tilda):
        """
        Calculate d_i_tilda

        Parameters
        ----------
        n : number of rows of the matrix X
        d_tilda : matrix previously calculated
        """

        minimo = np.array([0.0 for l in range(n)])
        for i in range(n):
            comparacon_minimo = np.array([0.0 for l in range(n)])
            for k in range(n):
                comparacon_minimo[k] = d_tilda[k, i]
            minimo[i] = np.amin(comparacon_minimo)
        return minimo

    def execute_karps_algorithm(self):
        """
        General function to be called outside the class in order to execute all the methods previoulsy defined

        Notes
        -----
        Tested for d > 1
        """

        # Karp's function begin
        """Calculate c_{i,j}"""
        n = self.matrix_X.shape[0]
        c = self.calculate_c_i_j(n, self.matrix_X, self.matrix_Y)

        """Calculate d_{k,i}"""
        d = self.calculate_d_k_i(n, c)

        """Calculate epsilon star"""
        (ass_A, epsilon_star) = self.calculate_epsilon_star(n, d)
        self.epsilon_star = epsilon_star

        """Calculate c_i_j_tilda """
        c_tilda = self.calculate_c_i_j_tilda(c, epsilon_star)

        """Calculate d_i_j_tilda"""
        d_tilda = self.calculate_d_k_i(n, c_tilda)

        """Calculate d_i_tilda"""
        d_i_tilda = self.calculate_d_i_tilda(n, d_tilda)

        """Turn to the original problem"""
        psi = (-1) * d_i_tilda
        self.psi = psi
        solution_karp = np.append(psi, np.array(epsilon_star))
        self.solution_karp = solution_karp

        # Karp's function end