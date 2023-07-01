import pandas as pd
import numpy as np
from numpy import linalg, asarray, savetxt, genfromtxt
from scipy.optimize import LinearConstraint, linprog
from pulp import *

'''
    File name: total_reparation.py
    Description: Script to implement the reparation methodology according to the following paper. https://arxiv.org/abs/1806.03195
    Author: Elena M. De Diego
    GitHub: https://github.com/emartindedi/OnlineDataRepair.git
    Package: OnlineDataRepair
    Python Version: 3.8 and 3.9
    License: No use withut explicit consention
'''


class Total_Reparation:

    r"""

    This class computes the reparation proccess in one dimension according to the following paper

    @InProceedings{pmlr-v97-gordaliza19a,
      title = 	 {Obtaining Fairness using Optimal Transport Theory},
      author =       {Gordaliza, Paula and Barrio, Eustasio Del and Fabrice, Gamboa and Loubes, Jean-Michel},
      booktitle = 	 {Proceedings of the 36th International Conference on Machine Learning},
      pages = 	 {2357--2365},
      year = 	 {2019},
      editor = 	 {Chaudhuri, Kamalika and Salakhutdinov, Ruslan},
      volume = 	 {97},
      series = 	 {Proceedings of Machine Learning Research},
      month = 	 {09--15 Jun},
      publisher =    {PMLR},
      pdf = 	 {http://proceedings.mlr.press/v97/gordaliza19a/gordaliza19a.pdf},
      url = 	 {https://proceedings.mlr.press/v97/gordaliza19a.html},
      abstract = 	 {In the fair classification setup, we recast the links between fairness and predictability in terms of probability metrics. We analyze repair methods based on mapping conditional distributions to the Wasserstein barycenter. We propose a Random Repair which yields a tradeoff between minimal information loss and a certain amount of fairness.}
    }

    Notes
    -----

    There are two alternatives of this proccess explained in the article that we implemented in two different functions
    """


    input_0 = None
    input_1 = None
    reparated_points_A_array = None
    reparated_points_B_array = None

    def __init__(self, input_0, input_1):
        """Initialize a new instance.

        Parameters
        ----------
        input_0 : pandas dataframe object in general dimension or a pandas series in dimension 1
        input_1 : pandas dataframe object in general dimension or a pandas series in dimension 1
        Notes
        ------
        Are the input variables for each class {0: minority class, 1: default class}
        """
        if not (isinstance(input_0, pd.DataFrame) or isinstance(input_0, pd.Series)):
            raise TypeError('The input_0 is not a required pandas dataframe object or pandas series object')
        if not (isinstance(input_1, pd.DataFrame) or isinstance(input_1, pd.Series)):
            raise TypeError('The input_1 is not a required pandas dataframe object or pandas series object')

        self.input_0 = input_0
        self.input_1 = input_1

    def cuadratic_cost_function(self, serie_1, serie_2, n_0, n_1):
        """
        Function to calculate the quadratic cost function C

        :param serie_1: pandas serie
        :param serie_2: pandas serie
        :param n_0: number of minority samples
        :param n_1: number of default samples
        :return: numpy matrix quadratic cost function C / c_{i,j} = || x_{0,i}-x_{1,j} ||^{2}
        """

        cost_mat = np.zeros((n_0, n_1)) # Create a null matrix of the dimension given

        for i in range(n_0):

            for j in range(n_1):

                cost_mat[i, j] = (np.linalg.norm(serie_1[i] - serie_2[j]))**2

        return cost_mat

    def transportation_problem(self, n_0, n_1, C):
        """
        Function that define the (14) optimization problem of the paper 'Obtaining Fairness using Optimal Transport Theory'.

        :param n_0: number of minority samples
        :param n_1: number of default samples
        :param C: cuadratic_cost_function
        :return: model object
        """
        # Objective function:
        cost_matrix = C
        # Model
        model = LpProblem("Transportation-Problem", LpMinimize)
        variable_names = [str(i) + ',' + str(j) for i in range(n_0) for j in range(n_1)]

        DV_variables = LpVariable.matrix("G", variable_names, cat="Integer")
        allocation = np.array(DV_variables).reshape(n_0, n_1)

        obj_func = lpSum(allocation * cost_matrix)
        model += obj_func

        # Constraints:
        for j in range(n_1):

            model += lpSum(allocation[i][j] for i in range(n_0)) == 1 / n_1, "Constraint one " + str(j)

        for i in range(n_0):

            model += lpSum(allocation[i][j] for j in range(n_1)) == 1 / n_0, "Constraint two " + str(i)

        # Non negative constrains:
        for i in range(n_0):

            for j in range(n_1):

                model += lpSum(allocation[i][j]) >= 0, "Non negative constraint " + str(i) + ',' + str(j)

        model.writeLP(r".\data\TransportationProblem.lp")

        return model

    def solve_transportation_problem(self, model):
        """
        Function that solve the (14) optimization problem of the paper 'Obtaining Fairness using Optimal Transport Theory'.
        :param model: model object
        :return: tuple (value of the objective function, dictionary with the variables as keys and solution values as
        values,array with the values of the variables)
        """
        solution_value_var = {}
        solution_array = []
        model.solve(PULP_CBC_CMD())

        # Decision Variables
        for v in model.variables():

            try:
                solution_value_var[v.name] = v.value()
                solution_array.append([v.value()])
            except:

                print("error couldnt find value")
        return (model.objective.value(), solution_value_var, np.array(solution_array))

    def define_total_repair_opcionA(self, serie_1, serie_2, n_0, n_1, pi_0, pi_1, gamma_opt):
        """
        Function that calculates opcion A, the repair points according to the papaer Obtaining Fairness section 4 - Option A.
        :param serie_1: pandas serie
        :param serie_2: pandas serie
        :param n_0: number of minority samples
        :param n_1: number of default samples
        :param: pi_0: float n_0/n_0+n_1
        :param: pi_1: float n_1/n_0+n_1
        :param: gamma_opt solution
        """
        repared_0 = np.zeros(n_0)
        repared_1 = np.zeros(n_1)
        # Reparacion X_{0}
        for i in range(n_0):
            a = 0
            for j in range(n_1):
                a += gamma_opt[i,j]*serie_2[j]
            repared_0[i] = (pi_0 * serie_1[i]) + (n_0*pi_1)*a

        # Reparacion X_{1}
        for j in range(n_1):
            b = 0
            for i in range(n_0):
                b += gamma_opt[i,j]*serie_1[i]
            repared_1[j] = (pi_1 * serie_2[j]) + (n_1*pi_0)*b

        return (repared_0, repared_1)

    def define_total_repair_opcionB(self, serie_1, serie_2, n_0, n_1, pi_0, pi_1, gamma_opt):
        """
        Function that calculates (15), the total repair points.
        :param serie_1: pandas serie
        :param serie_2: pandas serie
        :param n_0: number of minority samples
        :param n_1: number of default samples
        :param: pi_0: float n_0/n_0+n_1
        :param: pi_1: float n_1/n_0+n_1
        :param: gamma_opt solution
        :return: tuple (np.array, all non negative values in a list)
        """
        repared = np.zeros((n_0, n_1))
        total_repared = []  # list with the positive elements
        for i in range(n_0):
            for j in range(n_1):
                if gamma_opt[i,:][0,j] > 0:
                    repared[i,j] = (pi_0 * serie_1[i]) + (pi_1 * serie_2[j])
                    total_repared.append(repared[i][j])
        return (repared, total_repared)

    def get_wasserstein_barycenter(self, serie_1, serie_2, n_0, n_1, repared_data, gamma_opt):
        """
        Calculate the Wasserstein barycenter.
        :param serie_1: pandas serie
        :param serie_2: pandas serie
        :param n_0: number of minority samples
        :param n_1: number of default samples
        :param: reparated_data: all points reparated
        :param: gamma_opt: solution to the optimization problem.
        :return: mu_{B,n} of the paper"""
        mu_0_n = 0
        mu_1_n = 0
        for i in range(n_0):
            mu_0_n = mu_0_n + ((1/n_0)*serie_1[i])
        for j in range(n_1):
            mu_1_n = mu_1_n + ((1 / n_1) * serie_2[j])
        barycenter = 0
        for i in range(n_0):
            for j in range(n_1):
                barycenter = barycenter + (gamma_opt[i,j]*(repared_data[i,j]))
        return barycenter

    def run_transportation_problem(self, input_min, input_def, n_des, n_fav):
        """
        Function that execute the transportation problem
        :param input_min: pandas serie
        :param input_def: pandas serie
        :param n_des: number of minority samples
        :param n_fav: number of default samples
        """

        # 5. Quadratic cost function
        self.cuadratic_cost_function(np.array(input_min), np.array(input_def), n_des, n_fav)
        # save to csv file
        cuadr_cost_matrix = asarray(
            self.cuadratic_cost_function(np.array(input_min), np.array(input_def), n_des, n_fav))
        savetxt(r'.\data\c_transp_prob.csv', cuadr_cost_matrix, delimiter=',')

        # 6. TRANSPORTATION PROBLEM
        C = genfromtxt(r'.\data\c_transp_prob.csv', delimiter=',')

        # 7.TRANSPORTATION PROBLEM
        modelo = self.transportation_problem(n_des, n_fav, C)

        # 8.TRANSPORTATION PROBLEM
        (optimo, dict_gamma, array_gamma) = self.solve_transportation_problem(modelo)
        gamma_opt = np.matrix(array_gamma).reshape(n_des, n_fav)
        savetxt(r'.\data\gamma_opt.csv', gamma_opt, delimiter=',')
        return gamma_opt

    def run_total_repair_option_A(self):
        """
        Function to calculate the reparation data with opcion A
        :param dataset: pandas dataframe object
        :param protected_variable: string with the protected attribute name
        :param minority_class: string with the value of the protected variable for the minority group
        :param default_class: string with the value of the protected variable for the default group
        :param attribute_legitimate: column in the dataset that it would be repared
        """

        continues = True

        if continues:

            try:
                # Define the number of samples of each class of the protected variable
                n_des = self.input_0.shape[0]
                n_fav = self.input_1.shape[0]

            except Exception as e:
                print(e)
                continues = False

        if continues:
            try:
                # 9. TOTAL REPARATION
                pi_0 = n_des / (n_des + n_fav)
                pi_1 = n_fav / (n_des + n_fav)
                # Option A
                reparated_points_A = self.define_total_repair_opcionA(np.array(self.input_0), np.array(self.input_1), n_des, n_fav, pi_0,
                                                                  pi_1, self.run_transportation_problem(self.input_0, self.input_1, n_des, n_fav))
                df0 = pd.DataFrame({'0': reparated_points_A[0]})
                df1 = pd.DataFrame({'1': reparated_points_A[1]})
                reparated_points_A_df = pd.concat([df0, df1], ignore_index=True, axis=1)
                self.reparated_points_A_array = reparated_points_A_df
                reparated_points_A_df.to_csv(r'.\data\reparated_points_a.csv', index=False)

            except Exception as e:
                print(e)
                continues = False

    def run_total_repair_option_B(self):
        """
        Function to calculate the reparation data with opcion B
        :param dataset: pandas dataframe object
        :param protected_variable: string with the protected attribute name
        :param minority_class: string with the value of the protected variable for the minority group
        :param default_class: string with the value of the protected variable for the default group
        :param attribute_legitimate: column in the dataset that it would be repared
        """

        continues = True

        if continues:

            try:
                # Define the number of samples of each class of the protected variable
                n_des = self.input_0.shape[0]
                n_fav = self.input_1.shape[0]
            except Exception as e:
                print(e)
                continues = False

        if continues:

            try:

                # 9. TOTAL REPARATION
                pi_0 = n_des / (n_des + n_fav)
                pi_1 = n_fav / (n_des + n_fav)

                # Option B
                (reparated_points_B, list_reparated_points_B) = self.define_total_repair_opcionB(np.array(self.input_0), np.array(self.input_1), n_des, n_fav,
                                                               pi_0, pi_1, self.run_transportation_problem(self.input_0, self.input_1, n_des, n_fav))
                df0 = pd.DataFrame({'0': list_reparated_points_B})
                df1 = pd.DataFrame({'1': list_reparated_points_B})
                reparated_points_B_df = pd.concat([df0, df1], ignore_index=True, axis=1)
                self.reparated_points_B_array = reparated_points_B_df
                reparated_points_B_df.to_csv(r'.\data\reparated_points_b.csv', index=False)

            except Exception as e:
                print(e)
                continues = False

    def total_reparation_process(self, group, opcion = 'A'):
        """
        Main method of tha class to call it from outside the python file.
        Depending on the option selected, it performs one methodology or the other

        :param opcion (string): by default is option A
        :param group (int): by default is group 0
        :return: tuple : (original values of the group, reparated values of the group choosen)
        """


        try:
            if opcion == 'A':

                if group == 0:

                    self.run_total_repair_option_A()
                    return (np.array(self.input_0), np.array(self.reparated_points_A_array.iloc[:, 0][:min(self.input_0.shape[0], self.input_1.shape[0])]))

                elif group == 1:

                    self.run_total_repair_option_A()
                    return (np.array(self.input_1), np.array(
                        self.reparated_points_A_array.iloc[:, 1][:self.input_1.shape[0]]))

                else:

                    raise ValueError("Group must be or 0 or 1")

            else:
                # Option B

                if group == 0:

                    self.run_total_repair_option_B()

                    return (np.array(self.input_0), np.array(self.reparated_points_B_array.iloc[:, 0][:min(self.input_0.shape[0], self.input_1.shape[0])]))

                elif group == 1:

                    self.run_total_repair_option_B()
                    return (np.array(self.input_1), np.array(
                        self.reparated_points_B_array.iloc[:, 1][:self.input_1.shape[0]]))

                else:

                    raise ValueError("Group must be or 0 or 1")


        except Exception as e:

            print(e)


