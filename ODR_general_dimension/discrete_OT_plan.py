import pandas as pd
import numpy as np
import sys
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from munkres import Munkres, print_matrix
import ot

'''
    File name: discrete_OT_interpolation.py
    Description: Script to run after the reparation process to reassign correctly the two arrays
    Author: Elena M. De Diego
    GitHub: https://github.com/emartindedi/OnlineDataRepair.git
    Package: OnlineDataRepair
    Python Version: 3.8 and 3.9
    License: No use withut explicit consention
'''

"""
@article{flamary2021pot,
  author  = {R{\'e}mi Flamary and Nicolas Courty and Alexandre Gramfort and Mokhtar Z. Alaya and Aur{\'e}lie Boisbunon and Stanislas Chambon and Laetitia Chapel and Adrien Corenflos and Kilian Fatras and Nemo Fournier and L{\'e}o Gautheron and Nathalie T.H. Gayraud and Hicham Janati and Alain Rakotomamonjy and Ievgen Redko and Antoine Rolet and Antony Schutz and Vivien Seguy and Danica J. Sutherland and Romain Tavenard and Alexander Tong and Titouan Vayer},
  title   = {POT: Python Optimal Transport},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {78},
  pages   = {1-8},
  url     = {http://jmlr.org/papers/v22/20-451.html}
}
"""


class Association_problem:

    r"""

    This class solves the association problem between the vector_X points and vector_Y points

    For now, its implemented for dimesion 1

    There exists two possible algorithms:
    - Auction Algorithm: Auction Algorithms Dimitri P. Bertsekas bertsekas@lids.mit.edu Laboratory for
        Information and Decision Systems Massachusetts Institute of Technology Cambridge, MA 02139, USA
    - Hungarian Algorithm: https://en.wikipedia.org/wiki/Hungarian_algorithm

    Notes
    -----

    Computational cost: O(n^{2}) and O(n^{3}) resp.

    """

    def __init__(self, matrix_X, matrix_Y):
        """Initialize a new instance.

        Parameters
        ----------
        matrix_X : n tuple of points in R^{d}.
        matrix_Y : n tuple of points in R^{d}.

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

    def cuadratic_cost_function_ot(self):
        """
        Function to calculate the quadratic cost function C
        Returns
        -------
        :return: numpy matrix quadratic cost function C / c_{i,j} = || x_{0,i}-x_{1,j} ||^{2}
        """

        cost_matrix = distance_matrix(x=self.matrix_X,y=self.matrix_Y,p=2)**2
        return cost_matrix

    def cuadratic_cost_function(self, set_0, set_1, n_0, n_1):
        """
        Function to calculate the quadratic cost function C

        :param set_0: pandas Dataframe
        :param set_1: pandas Dataframe
        :param n_0: number of minority samples
        :param n_1: number of default samples
        :return: numpy matrix quadratic cost function C / c_{i,j} = || x_{0,i}-x_{1,j} ||^{2}
        """

        cost_mat = np.zeros((n_0, n_1))  # Create a null matrix of the dimension given

        for i in range(n_0):

            for j in range(n_1):
                cost_mat[i, j] = (np.linalg.norm(set_0.iloc[i, :] - set_1.iloc[j, :])) ** 2

        return cost_mat

    def optimal_linear_assignment(self):
        """
        Solve the linear sum assignment problem.

        The linear sum assignment problem is also known as minimum weight matching in bipartite graphs.
        A problem instance is described by a matrix C, where each C[i,j] is the cost of matching vertex
        i of the first partite set (a “worker”) and vertex j of the second set (a “job”). The goal is
        to find a complete assignment of workers to jobs of minimal cost.

        :return: An array of row indices and one of corresponding column indices giving the optimal assignment.
        """

        n_0, n_1 = self.vector_X.shape[0], self.vector_Y.shape[0]
        cost = self.cuadratic_cost_function(self.vector_X, self.vector_Y, n_0, n_1)

        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
        row_ind, col_ind = linear_sum_assignment(cost)

        return col_ind

    def run_hungarian_w_munkres(self):

        """

        https://pypi.org/project/munkres/1.0.5.4/

        :return: list with the solution
        """

        n_0, n_1 = self.vector_X.shape[0], self.vector_Y.shape[0]
        cost = self.cuadratic_cost_function(self.vector_X, self.vector_Y, n_0, n_1)

        m = Munkres()

        # Pad the cost matrix to make it cuadratic
        matrix = m.pad_matrix(cost, pad_value=0)

        indexes = m.compute(matrix)

        return indexes

    def arrange_reparated_data(self):
        """
        Method to manage the class.
        If the order obtained with the method optimal_linear_assignment


        :return: tuple ():
        """

        try:

            order = self.optimal_linear_assignment()  # numpy array solution of optimal_linear_assignment

            vector_X_mod = np.zeros(self.vector_X.shape)
            vector_Y_mod = np.zeros(self.vector_Y.shape)

            if (order == np.arange(self.vector_X.shape[0])).all():
                # If we are here means that the assignation is well done but we dont have the original vector
                # ordered by its values
                order_vector_X = self.vector_X.argsort().argsort()

                for i in range(self.vector_X.shape[0]):
                    l = order_vector_X == i
                    vector_X_mod[i] = self.vector_X[l]
                    vector_Y_mod[i] = self.vector_Y[l]

                return (vector_X_mod, vector_Y_mod)

            else:

                ind_munkres = self.run_hungarian_w_munkres()

                for i in range(self.vector_X.shape[0]):

                    l = ind_munkres[i][1]

                    vector_X_mod[i] = self.vector_X[i]
                    vector_Y_mod[i] = self.vector_Y[l]

                return (vector_X_mod, vector_Y_mod)

        except Exception as e:

            print(e)

    def ass_prob_solved_ot(self):
        """

        Returns
        -------

        """
        try:

            cost = self.cuadratic_cost_function_ot()
            w = np.ones(self.matrix_X.shape[0])
            Pn, log = ot.emd(w, w, cost, log=True, numItermax=100000000)
            ccc = ot.solve(cost, w, w)
            new_index = (Pn @ np.arange(self.matrix_X.shape[0])).astype('int32')

            print("new_index", new_index)

            matrix_Y_sort = self.matrix_Y.copy()
            matrix_Y_sort['new_index'] = new_index
            matrix_Y_sort.sort_values(by=['new_index'], inplace=True)
            matrix_Y_sort.reset_index(inplace=True)
            matrix_Y_sort.drop(columns=['new_index', 'index'], inplace=True)

            return (self.matrix_X, matrix_Y_sort)

        except Exception as e:

            print(e)