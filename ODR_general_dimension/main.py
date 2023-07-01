from OnlineDataRepair.ODR_general_dimension.total_reparation import Total_Reparation
from OnlineDataRepair.ODR_general_dimension.discrete_OT_plan import Association_problem
from OnlineDataRepair.ODR_general_dimension.karps_algorithm import Karp
from OnlineDataRepair.ODR_general_dimension.OT_interpolation import OT_interpolation
from OnlineDataRepair.ODR_general_dimension.utils import *

import pandas as pd
import numpy as np

'''
    File name: main.py
    Description: Main function that manage all of the package modules
    Author: Elena M. De Diego
    GitHub: https://github.com/emartindedi/OnlineDataRepair.git
    Package: OnlineDataRepair
    Python Version: 3.8 and 3.9
    License: No use withut explicit consention
'''



def run_OT_extension_gen_dim(input_0, input_1, x, dim = 1, grupo = 0, opcion_rep = 'A', rtol=1e-6, maxiter=100, default_step_size=0.2):

    """
    Function that manages all classes in the OT_extension package. It should be the one called to calculate
    the reparated version of a new point x in R^{d}
    :param input_0: X matrix for class 0
    :param input_1: X matrix for class 1
    :param x: new point to calculate the reparated version
    :param dim: the dimension of X without the protected variable
    :param grupo: possible variables {0, 1}
    :param: opcion_rep: option to reparate totally the biased data
    :param rtol: tolerance (default = 1e-6)
    :param maxiter: maximum of iterations (default = 100)
    :param default_step_size: step size (default = 0.2)
    :return: image of x by the function T
    """

    continues = True

    if not isinstance(x, (int, float, pd.Series)):
        raise TypeError('the element x is not a proper type of object: int, float or pandas Series')

    if not isinstance(dim, int):
        raise TypeError('The element dim is not an integer')

    if not isinstance(grupo, int):
        raise TypeError('The element grupo is not an integer')

    if not isinstance(opcion_rep, str):
        raise TypeError('The option valid is not an string')


    if not(grupo == 0 or grupo == 1):

        continues = False
        raise ValueError("Grupo value must be or 0 or 1")

    else:

        continues = True

        if grupo == 0:
            # We are reparing a element of the minority group

            if dim >= 1:

                if dim == 1:

                    if continues:

                        try:
                            """ Reparation process """
                            # total_reparation

                            rep_tot = Total_Reparation(input_0, input_1)
                            (vector_X, vector_Y) = rep_tot.total_reparation_process(group = grupo, opcion=opcion_rep)

                            print("Original vector of the class {} is {}".format(grupo, vector_X))
                            print("Reparated vector of the class {} with the option {} is {}".format(grupo, opcion_rep,vector_Y))

                        except Exception as e:

                                print(e)
                                continues = False

                    if continues:
                        """Verify the shape of the original and reparated. NECESSARY CONDITION"""
                        try:

                            if vector_X.shape[0] != vector_Y.shape[0]:
                                continues = False
                                raise ValueError("The length of input_0 and input_1 are not the same. Requirement.")
                            else:
                                n = vector_X.shape[0]
                                continues = True

                        except Exception as e:

                                print(e)
                                continues = False

                    if continues:

                        try:
                            """ Optimal assignment """
                            ass_prob = Association_problem(vector_X, vector_Y)
                            (vector_X_mod, vector_Y_mod) = ass_prob.arrange_reparated_data()
                            print("vector_X_mod", vector_X_mod)
                            print("vector_Y_mod", vector_Y_mod)

                        except Exception as e:

                            print(e)
                            continues = False

                    if continues:

                        """ Verificar la condicion de cíclicamente monotono """
                        # utils

                        try:

                            bool_cicli_mon = cyclically_monotone(vector_X_mod, vector_Y_mod)

                        except Exception as e:

                            print(e)
                            continues = False

                    if continues:

                        """ Karps' algorithm """
                        # karps_algorithm

                        try:

                            karp = Karp(vector_X_mod, vector_Y_mod)
                            karp.execute_karps_algorithm()
                            epsilon_zero = (karp.epsilon_star) / 2  # epsilon_zero
                            print("EPSILON ZERO", epsilon_zero)

                        except Exception as e:
                            print(e)
                            continues = False

                    if continues:

                        """ Interpolation algorithm """
                        # OT_interpolation

                        try:

                            if epsilon_zero <= 0:

                                print("The parameter epsilon computed by the Karps algorithm must be extrictly positive")
                                continues = False
                                raise ValueError("Epsilon negative or zero")

                            else:

                                interpolation = OT_interpolation(vector_X_mod, vector_Y_mod)
                                proximal_x = interpolation.function_OT_interpolation(karp.psi, epsilon_zero, x, rtol=1e-6, maxiter=100, default_step_size=0.2)

                        except Exception as e:

                            print(e)
                            continues = False

                    if continues:

                        """ Final steo """
                        # Give the reparated value for the new point

                        try:

                            T_x_new = (1/epsilon_zero)*(x - proximal_x)
                            print("-"*30)
                            print("T_x_new", T_x_new)
                            print("-" * 30)

                        except Exception as e:

                            print(e)
                            continues = False

                else:

                    # GENERAL DIMENSION (dim > 1)

                    if continues:

                        try:
                            """ Reparation process """
                            #total_reparation

                            rep_tot = Total_Reparation(input_0, input_1)
                            (matrix_X, matrix_Y) = rep_tot.total_reparation_process(group=grupo, opcion=opcion_rep)

                            print("Original matrix ", matrix_X.shape, "of the class", grupo)
                            print("Reparated matrix ", matrix_Y.shape, "of the class", grupo, "with the option", opcion_rep)
                            #print("Original matrix of the class {} is {}".format(grupo, matrix_X))
                            #print("Reparated matrix of the class {} with the option {} is {}".format(grupo, opcion_rep,
                            #                                                                         matrix_Y))

                        except Exception as e:

                            print(e)
                            continues = False

                    if continues:

                        """Verify the shape of the original and reparated. NECESSARY CONDITION"""
                        try:

                            if matrix_X.shape != matrix_Y.shape:
                                continues = False
                                raise ValueError("The shape of input_0 and input_1 are not the same. Requirement.")
                            else:
                                n = matrix_X.shape[0]
                                continues = True

                        except Exception as e:

                            print(e)
                            continues = False

                    if continues:

                        try:
                            """ Optimal assignment """
                            ass_prob = Association_problem(matrix_X, matrix_Y)
                            (matrix_X_mod, matrix_Y_mod) = ass_prob.ass_prob_solved_ot()
                            print("matrix_X_mod", matrix_X_mod)
                            print("matrix_Y_mod", matrix_Y_mod)

                        except Exception as e:

                            print(e)
                            continues = False

                    if continues:

                        """ Verificar la condicion de cíclicamente monotono """
                        # utils

                        try:

                            bool_cicli_mon = cyclically_monotone_general_dimension(matrix_X_mod, matrix_Y_mod)

                        except Exception as e:

                            print(e)
                            continues = False

                    if continues:

                        """ Karps' algorithm """
                        # karps_algorithm

                        try:

                            karp = Karp(matrix_X_mod, matrix_Y_mod)
                            karp.execute_karps_algorithm()
                            epsilon_zero = (karp.epsilon_star) / 2  # epsilon_zero
                            print("Epsilon zero: ", epsilon_zero)

                        except Exception as e:
                            print(e)
                            continues = False

                    if continues:

                        """ Interpolation algorithm """
                        # OT_interpolation

                        try:

                            if epsilon_zero <= 0:

                                print(
                                    "The parameter epsilon computed by the Karps algorithm must be extrictly positive")
                                continues = False
                                raise ValueError("Epsilon negative or zero")

                            else:

                                interpolation = OT_interpolation(matrix_X_mod, matrix_Y_mod)
                                proximal_x = interpolation.function_OT_interpolation(karp.psi, epsilon_zero, x,
                                                                                     rtol=1e-6, maxiter=100,
                                                                                     default_step_size=0.2)

                        except Exception as e:

                            print(e)
                            continues = False

                    if continues:

                        """ Final steo """
                        # Give the reparated value for the new point

                        try:

                            T_x_new = (1 / epsilon_zero) * (x.subtract(proximal_x, fill_value=0))
                            print("-" * 30)
                            print("T_x_new", T_x_new)
                            print("-" * 30)


                        except Exception as e:

                            print(e)
                            continues = False

            else:

                raise ValueError("Dimension value must be greater or equal to 1")

        else:
            # Grupo == 1
            # We are reparing a element of the default group

            if dim >=1:

                if dim == 1:

                    if continues:

                        try:
                            """ Reparation process """
                            # total_reparation

                            rep_tot = Total_Reparation(input_0, input_1)
                            (vector_X, vector_Y) = rep_tot.total_reparation_process(group=grupo, opcion=opcion_rep)

                            print("Original vector of the class {} is {}".format(grupo, vector_X))
                            print("Reparated vector of the class {} with the option {} is {}".format(grupo, opcion_rep,
                                                                                                     vector_Y))

                        except Exception as e:

                                print(e)
                                continues = False

                    if continues:
                        try:
                            if vector_X.shape[0] != vector_Y.shape[0]:
                                continues = False
                                raise ValueError("The length of input_0 and input_1 are not the same. Requirement.")
                            else:
                                n = vector_X.shape[0]
                                continues = True

                        except Exception as e:

                                print(e)
                                continues = False

                    if continues:

                        try:
                            """ Optimal assignment """
                            # Rearrange to order the values

                            ass_prob = Association_problem(vector_X, vector_Y)
                            (vector_X_mod, vector_Y_mod) = ass_prob.arrange_reparated_data()
                            print("vector_X_mod", vector_X_mod)
                            print("vector_Y_mod", vector_Y_mod)

                        except Exception as e:

                            print(e)
                            continues = False

                    if continues:

                        """ Verificar la condicion de cíclicamente monotono """
                        # utils

                        try:

                            bool_cicli_mon = cyclically_monotone(vector_X_mod, vector_Y_mod)

                        except Exception as e:

                            print(e)
                            continues = False

                    if continues:

                        """ Karps' algorithm """
                        # karps_algorithm

                        try:

                            karp = Karp(vector_X_mod, vector_Y_mod)
                            karp.execute_karps_algorithm()
                            epsilon_zero = (karp.epsilon_star) / 2  # epsilon_zero

                        except Exception as e:
                            print(e)
                            continues = False

                    if continues:

                        """ Interpolation algorithm """
                        # OT_interpolation

                        try:

                            if epsilon_zero <= 0:

                                print("The parameter epsilon computed by the Karps algorithm must be extrictly positive")
                                continues = False
                                raise ValueError("Epsilon negative or zero")

                            else:

                                interpolation = OT_interpolation(vector_X_mod, vector_Y_mod)
                                proximal_x = interpolation.function_OT_interpolation(karp.psi, epsilon_zero, x, rtol=1e-6, maxiter=100, default_step_size=0.2)

                        except Exception as e:

                            print(e)
                            continues = False

                    if continues:

                        """ Final steo """
                        # Give the reparated value for the new point

                        try:

                            T_x_new = (1/epsilon_zero)*(x - proximal_x)
                            print("-"*30)
                            print("T_x_new", T_x_new)
                            print("-" * 30)

                        except Exception as e:

                            print(e)
                            continues = False

                else:

                    # GENERAL DIMENSION (dim > 1)

                    if continues:

                        try:
                            """ Reparation process """
                            # total_reparation

                            rep_tot = Total_Reparation(input_0, input_1)
                            (matrix_X, matrix_Y) = rep_tot.total_reparation_process(group=grupo, opcion=opcion_rep)

                            print("Original matrix ", matrix_X.shape, "of the class", grupo)
                            print("Reparated matrix ", matrix_Y.shape, "of the class", grupo, "with the option",
                                  opcion_rep)
                            # print("Original matrix of the class {} is {}".format(grupo, matrix_X))
                            # print("Reparated matrix of the class {} with the option {} is {}".format(grupo, opcion_rep,
                            #                                                                         matrix_Y))

                        except Exception as e:

                            print(e)
                            continues = False

                    if continues:

                        """Verify the shape of the original and reparated. NECESSARY CONDITION"""
                        try:

                            if matrix_X.shape != matrix_Y.shape:
                                continues = False
                                raise ValueError("The shape of input_0 and input_1 are not the same. Requirement.")
                            else:
                                n = matrix_X.shape[0]
                                continues = True

                        except Exception as e:

                            print(e)
                            continues = False

                    if continues:

                        try:
                            """ Optimal assignment """
                            ass_prob = Association_problem(matrix_X, matrix_Y)
                            (matrix_X_mod, matrix_Y_mod) = ass_prob.ass_prob_solved_ot()
                            print("matrix_X_mod", matrix_X_mod)
                            print("matrix_Y_mod", matrix_Y_mod)

                        except Exception as e:

                            print(e)
                            continues = False

                    if continues:

                        """ Verificar la condicion de cíclicamente monotono """
                        # utils

                        try:

                            bool_cicli_mon = cyclically_monotone_general_dimension(matrix_X_mod, matrix_Y_mod)

                        except Exception as e:

                            print(e)
                            continues = False

                    if continues:

                        """ Karps' algorithm """
                        # karps_algorithm

                        try:

                            karp = Karp(matrix_X_mod, matrix_Y_mod)
                            karp.execute_karps_algorithm()
                            epsilon_zero = (karp.epsilon_star) / 2  # epsilon_zero
                            print("Epsilon zero: ", epsilon_zero)

                        except Exception as e:
                            print(e)
                            continues = False

                    if continues:

                        """ Interpolation algorithm """
                        # OT_interpolation

                        try:

                            if epsilon_zero <= 0:

                                print(
                                    "The parameter epsilon computed by the Karps algorithm must be extrictly positive")
                                continues = False
                                raise ValueError("Epsilon negative or zero")

                            else:

                                interpolation = OT_interpolation(matrix_X_mod, matrix_Y_mod)
                                proximal_x = interpolation.function_OT_interpolation(karp.psi, epsilon_zero, x,
                                                                                     rtol=1e-6, maxiter=100,
                                                                                     default_step_size=0.2)

                        except Exception as e:

                            print(e)
                            continues = False

                    if continues:

                        """ Final steo """
                        # Give the reparated value for the new point

                        try:

                            T_x_new = (1 / epsilon_zero) * (x.subtract(proximal_x, fill_value=0))
                            print("-" * 30)
                            print("T_x_new", T_x_new)
                            print("-" * 30)


                        except Exception as e:

                            print(e)
                            continues = False

            else:

                raise ValueError("Dimension value must be greater or equal to 1")