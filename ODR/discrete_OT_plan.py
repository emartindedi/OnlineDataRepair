import numpy as np
import sys
from scipy.optimize import linear_sum_assignment
from munkres import Munkres, print_matrix


'''
    File name: discrete_OT_interpolation.py
    Description: Script to run after the reparation process to reassign correctly the two arrays
    Author: Elena M. De Diego
    GitHub: https://github.com/emartindedi/OnlineDataRepair.git
    Package: OnlineDataRepair
    Python Version: 3.8 and 3.9
    License: No use withut explicit consention
'''



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

    def __init__(self, vector_X, vector_Y):
        """Initialize a new instance.

        Parameters
        ----------
        vector_X : n tuple of points in R^{d}. For now its implemented with d=1
        vector_Y : n tuple of points in R^{d}. For now its implemented with d=1

        Notes
        ------
        Both of them needs to be an instance of numpy array
        """
        if not isinstance(vector_X, (np.ndarray, np.generic)):
            raise TypeError('the vector X is not a numpy array')
        if not isinstance(vector_Y, (np.ndarray, np.generic)):
            raise TypeError('the vector Y is not a numpy array')
        self.vector_X = vector_X
        self.vector_Y = vector_Y

    def cuadratic_cost_function(self, serie_1, serie_2, n_0, n_1):
        """Function to calculate the cuadratic cost function C
        :param serie_1: pandas serie
        :param serie_2: pandas serie
        :param n_0: lenght of the numpy vector serie_0
        :param: n_1: lenght of the numpy vector serie_1
        :return: numpy matrix cuadratic cost function C / c_{i,j} = || x_{0,i}-x_{1,j} ||^{2}
        """
        cost_mat = np.zeros((n_0, n_1))  # Create a null matrix of the dimension given

        for i in range(n_0):
            for j in range(n_1):
                cost_mat[i, j] = (np.linalg.norm(serie_1[i] - serie_2[j])) ** 2

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

            continues = False