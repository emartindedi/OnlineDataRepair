from OnlineDataRepair.ODR_general_dimension.main import run_OT_extension_gen_dim, cyclically_monotone_general_dimension
from OnlineDataRepair.ODR_general_dimension.utils import *

import time

'''
    File name: example_general_dimension.py
    Description: Script to run a general dimension example with simulated data
    Author: Elena M. De Diego
    GitHub: https://github.com/emartindedi/OnlineDataRepair.git
    Package: OnlineDataRepair
    Python Version: 3.8 and 3.9
    License: No use withut explicit consention
'''



if __name__ == "__main__":

    start = time.time()  # units: seconds

    """ Simulated data """

    dim = 20
    n_0 = 5
    n_1 = 5

    #seed = 1999
    #np.random.seed(1999)

    data = simulated_data_general_dimension(n_0, n_1, dim)
    print(data)
    not_protected = [col for col in data.columns if col != 'protected']

    input_min = data[data['protected'] == 0]
    input_def = data[data['protected'] == 1]

    input_0 = input_min[not_protected]
    input_1 = input_def[not_protected]

    x_0 = pd.Series([round(np.random.normal(2,4),1) for ele in range(dim)])

    run_OT_extension_gen_dim(input_0, input_1, x_0, dim=dim, grupo=0, opcion_rep='A') #dim-1
    print("x_0", x_0)

    end = time.time()
    print("TIME EXECUTION:", end - start)