import numpy as np
from OnlineDataRepair.ODR.utils import *

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


    def function_phi(self, psi_array, x):

        """

        Function to calculate the (3.3) function. It returns the image of x

        $$\Tilde{\varphi_{n}}(x) = \max_{1\leq j \leq n} \{ \langle x, \tilde{x_{j}} \rangle - \psi_{j}  \}$$

        :param: psi_array: numpy array calculated with Karps' algorithm
        :param: x: value to evaluate the function phi
        :return: the evaluation of the function \Tilde{\varphi_{n}}(x) of the given point x
        """

        maximo = np.zeros(len(self.vector_X)) # initialized

        for j in range(len(self.vector_X)):

            maximo[j] = np.inner(np.array(x), self.vector_Y[j]) - psi_array[j]

        return np.amax(maximo)

    def grad_function_phi_n(self, psi_array, x):

        """
        Function to calculate the subgradient of function_phi

        $$\Tilde{\varphi_{n}}(x) = \max_{1\leq j \leq n} \{ f_{j} (x) \} =
        \max_{1\leq j \leq n} \{ \langle x, \tilde{x_{j}} \rangle - \psi_{j}  \}$$


        $$\partial \Tilde{\varphi_{n}}(x) = \text{convex} \bigcup \{\partial f_{j} (x) | f_{j} (x) = \Tilde{\varphi_{n}}(x)\}$$

        :param: psi_array (calculated with Karps algorithm)
        :param: x (float to evaluate)
        :return:
        """

        f = np.zeros(len(self.vector_X))

        for j in range(len(self.vector_X)):

            f[j] = np.inner(np.array(x), self.vector_Y[j]) - psi_array[j]

        pos_max = np.flatnonzero(f == np.max(f))

        convex_hull_points = self.vector_Y[pos_max]

        return convex_hull_points

    def gradient_descent_algorithm(self, psi_array, optimal_smothing_value, x_eval, y0=0.0, rtol=1e-6,
                                   maxiter=1000, default_step_size=0.2):
        """
        Function that implement the Subgradient Descent algorithm
        Dimension d = 1

        :param psi_array : numpy vector calculated with Karps' algorithm
        :param: optimal_smothing_value: positive value calculated with Karps' algorithm
        :param: x_eval: point where we want to evaluate the interpolation function
        :param y0 : array-like
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
        num_points_malla = 100 # antes: 50

        # Interval of x_eval
        pos_x_eval = [0, self.vector_X.shape[0] - 1]

        for it in range(maxiter):
            print("-" * 5)
            print("ITERATION: ", it)

            # .. step 1 ..
            # Select a element of the subgradient
            print("yt", yt)
            grad_ft_prov = self.grad_function_phi_n(psi_array, yt)
            print("grad_ft_prov", grad_ft_prov)
            grad_ft = grad_ft_prov[0] + (1 / optimal_smothing_value) * (yt - x_eval)
            print("grad_ft", grad_ft)

            # .. step 2 ..
            # Compute the subgradient descent algorithm step
            yt = yt - (default_step_size * grad_ft)
            print("yt", yt)

            # .. step 3 ..
            # Projected
            #if (round(x_eval,5) >= round(self.vector_X[0],5) and round(x_eval,5) <= round(self.vector_X[1],5)):
            #    malla = np.linspace(
            #        self.vector_X[0] - (optimal_smothing_value * self.vector_Y[0]),
            #        self.vector_X[1], num=num_points_malla)
            #else:
            #    malla = np.linspace(self.vector_X[pos_x_eval[0]], self.vector_X[pos_x_eval[1]], num=num_points_malla)

            #dist = []
            #for i in range(num_points_malla):
            #    dist.append(np.linalg.norm(np.array([[yt]]) - malla[i], axis=1))  # Euclidean distance
            #min_distance_index = np.argmin(dist)  # Find index of minimum distance
            #closest_vector = malla[min_distance_index]  # Get vector having minimum distance
            #yt = closest_vector
            #print("yt_proj", yt)

            # .. step 4 ..
            # Check
            euclidean_norm = lambda x, y: np.linalg.norm(np.array([x - y]), ord=2) ** 2  # d>1 add: axis=1
            eval_fun[it + 1] = self.function_phi(psi_array, yt) + (
                        (1 / (2 * optimal_smothing_value)) * (euclidean_norm(yt, x_eval)))
            print("eval_fun", eval_fun[it + 1])

            # .. step 5 ..
            default_step_size = optimal_smothing_value / (it + 1)

            # Stop conditions
            if eval_fun[it + 1] < rtol:
                success = True
                break
            if (it >= 1) and (eval_fun[it + 1] > eval_fun[it]):
                success = True
                break
            if abs(grad_ft) < rtol:
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

        :param x_eval: point to evaluate the interpolation function
        :return: image T(x_eval)

        """

        continues = True

        if continues:

            try:

                y0 = self.vector_X[0]
                (ok, result) = self.gradient_descent_algorithm(psi_array, epsilon_zero, x_eval, y0,
                                                          rtol=rtol, maxiter=maxiter, default_step_size=default_step_size)
                print("result function_OT_interpolation", result)
                return result

            except Exception as e:

                print(e)
                continues = False
                return False