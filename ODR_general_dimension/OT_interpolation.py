import numpy as np
from numpy import linalg as LA
from OnlineDataRepair.ODR_general_dimension.utils import *

'''
    File name: OT_interpolation.py
    Description: Script to compute the image of a element by the intepolation function T. 
    Author: Elena M. De Diego
    GitHub: https://github.com/emartindedi/OnlineDataRepair.git
    Package: OnlineDataRepair
    Python Version: 3.8 and 3.9
    License: No use withut explicit consention
'''


class OT_interpolation:

    r"""

    A Consistent Extension of Discrete Optimal Transport Maps


    $$\Tilde{\varphi_{n}}(x) = \max_{1\leq j \leq n} \{ f_{j} (x) \} = \max_{1\leq j \leq n} \{ \langle x, \tilde{x_{j}} \rangle - \psi_{j}  \}$$

    $$T(x) = \nabla \varphi_{\epsilon_{0}}(x) =
        \frac{1}{\epsilon_{0}} (x - \mathrm{prox}_{\epsilon_{0} \Tilde{\varphi_{n}}}(x))$$
    $$prox_{\sigma f}(x) = \sup_{y} \left\{ f(y) - \frac{1}{2\sigma} \| y-x \|_2^2 \right\}$$

    Notes
    ----

    Solved with the subgradient descent algorithm

    """

    def __init__(self, matrix_X, matrix_Y):

        """Initialize a new instance.

        Parameters
        ----------
        matrix_X : n tuple of points in R^{d}. (pandas Dataframe)
        matrix_Y : n tuple of points in R^{d}. (pandas Dataframe)

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


    def function_phi(self, psi_array, x):

        """

        Function to calculate the (3.3) function. It returns the image of x

        $$\Tilde{\varphi_{n}}(x) = \max_{1\leq j \leq n} \{ \langle x, \tilde{x_{j}} \rangle - \psi_{j}  \}$$

        :param: psi_array: numpy array calculated with Karps' algorithm
        :param: x: value to evaluate the function phi (pandas Series object)
        :return: the evaluation of the function \Tilde{\varphi_{n}}(x) of the given point x
        """

        maximo = np.zeros(self.matrix_X.shape[0])

        for j in range(self.matrix_X.shape[0]):

            maximo[j] = np.inner(x, self.matrix_Y.iloc[j,:]) - psi_array[j]

        return np.amax(maximo)

    def grad_function_phi_n(self, psi_array, x):

        """
        Function to calculate the subgradient of function_phi

        $$\Tilde{\varphi_{n}}(x) = \max_{1\leq j \leq n} \{ f_{j} (x) \} =
        \max_{1\leq j \leq n} \{ \langle x, \tilde{x_{j}} \rangle - \psi_{j}  \}$$


        $$\partial \Tilde{\varphi_{n}}(x) = \text{convex} \bigcup \{\partial f_{j} (x) | f_{j} (x) = \Tilde{\varphi_{n}}(x)\}$$

        :param: psi_array (calculated with Karps algorithm)
        :param: x : value to evaluate the function grad_function_phi_n (pandas Series object)
        :return: the evaluation of the function \partial \Tilde{\varphi_{n}}(\boldsymbol{x}}) of the given point x
        """

        f = np.zeros(self.matrix_Y.shape[0])

        for j in range(self.matrix_Y.shape[0]):

            f[j] = np.inner(x, self.matrix_Y.iloc[j,:]) - psi_array[j]

        pos_max = np.argmax(f)
        grad = self.matrix_Y.iloc[pos_max, :]

        return grad

    def gradient_descent_algorithm(self, psi_array, optimal_smothing_value, x_eval, y0, rtol=1e-6,
                                   maxiter=1000, default_step_size=0.2):

        """
        Function that implement the Subgradient Descent algorithm
        Dimension d > 1

        :param psi_array : numpy vector calculated with Karps' algorithm
        :param: optimal_smothing_value: positive value calculated with Karps' algorithm
        :param: x_eval: point where we want to evaluate the interpolation function (pandas Series)
        :param y0 : pandas Series
            Initial guess
        :param maxiter : int
            Maximum number of iterations.
        :param default_step_size : float
            Starting value for the line-search procedure.
        :return : tuple with a boolean indicating if the sgd algorithm converged or not and the result
        """

        # .. step 0 ..
        # Initial values

        yt = y0 # inital point
        success = False
        default_step_size = optimal_smothing_value
        eval_fun = {}
        num_points_malla = 100
        # Interval of x_eval
        #pos_x_eval = [0, self.vector_X.shape[0] - 1]

        for it in range(maxiter):
            print("-" * 5)
            print("ITERATION: ", it)

            # .. step 1 ..
            # Select a element of the subgradient
            print("yt", yt)
            grad_ft_prov = self.grad_function_phi_n(psi_array, yt)
            print("grad_ft_prov", grad_ft_prov)
            #grad_ft = grad_ft_prov + (1 / optimal_smothing_value) * (yt - x_eval)
            #print(yt.subtract(x_eval, fill_value=0).reset_index(drop=True))
            grad_ft = grad_ft_prov + (1 / optimal_smothing_value) * (yt.reset_index(drop=True).subtract(x_eval.reset_index(drop=True), fill_value=0))
            print("grad_ft", grad_ft)

            # .. step 2 ..
            # Compute the subgradient descent algorithm step
            yt = yt - (default_step_size * grad_ft)
            print("yt", yt)


            # .. step 3 ..
            # Check
            euclidean_norm = lambda x, y: np.linalg.norm(np.array([x.subtract(y, fill_value=0)]), ord=2) ** 2  # d>1 add: axis=1
            eval_fun[it + 1] = self.function_phi(psi_array, yt) + (
                        (1 / (2 * optimal_smothing_value)) * (euclidean_norm(yt, x_eval)))
            print("eval_fun", eval_fun[it + 1])

            # .. step 4 ..
            default_step_size = default_step_size / (it + 1)
            print("default_step_size", default_step_size)

            # Stop conditions
            if eval_fun[it + 1] < rtol:
                success = True
                break

            if (it >= 2) and (eval_fun[it + 1] > eval_fun[it]):
                success = True
                break

            if np.all(np.abs(grad_ft) < rtol):
            #if LA.norm(grad_ft, 1) < rtol:
                # np.all(np.abs(diff) <= rtol) general dimension
                success = True
                break



        return (success, yt)


    def function_OT_interpolation(self, psi_array, epsilon_zero, x_eval, rtol=1e-5, maxiter=100, default_step_size=0.2):

        """
        Main function to execute from outside the class

        $$T(x) = \nabla \varphi_{\epsilon_{0}}(x) =
        \frac{1}{\epsilon_{0}} (x - \mathrm{prox}_{\epsilon_{0} \Tilde{\varphi_{n}}}(x))$$

        $$prox_{\sigma f}(x) = \sup_{y} \left\{ f(y) -
            \frac{1}{2\sigma} \| y-x \|_2^2 \right\}$$

        :param x_eval: point to evaluate the interpolation function (pandas series)
        :return: image T(x_eval)

        """

        continues = True

        if continues:

            try:

                y0 = self.matrix_Y.iloc[0,:]
                (ok, result) = self.gradient_descent_algorithm(psi_array, epsilon_zero, x_eval, y0,
                                                          rtol=rtol, maxiter=maxiter, default_step_size=default_step_size)
                print("result function_OT_interpolation", result)
                return result

            except Exception as e:

                print(e)
                continues = False
                return False