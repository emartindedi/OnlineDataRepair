from OnlineDataRepair.ODR.main import run_OT_extension
from OnlineDataRepair.ODR.utils import *

import time

'''
    File name: example_one_dimension.py
    Description: Script to run a one dimension example with simulated data
    Author: Elena M. De Diego
    GitHub: https://github.com/emartindedi/OnlineDataRepair.git
    Package: OnlineDataRepair
    Python Version: 3.8 and 3.9
    License: No use withut explicit consention
'''

if __name__ == "__main__":

    start = time.time() # units: seconds

    """ Simulated data """
    # generation_data

    num_min = 5
    num_may = 6
    n = min(num_min, num_may)

    seed = 1999
    data = simulated_data(num_min, num_may)

    (dataset_minority, dataset_default) = divide_by_protected(data, 'sex', 'Integer', '0', '1')

    input_min = extract_column(dataset_minority, 'education').drop_duplicates()
    input_def = extract_column(dataset_default, 'education').drop_duplicates()

    print("input_min", input_min)
    print("input_def", input_def)

    #x_0 = random.uniform(2.43124471, 4.36503972)
    x_0 = 6.38412466
    #print("x grupo 0", x_0)

    run_OT_extension(input_min, input_def, x_0, dim = 1, grupo= 0, opcion_rep='B')
    print("x_0", x_0)

    end = time.time()
    print("TIME EXECUTION:", end - start)