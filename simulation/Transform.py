__author__ = 'bhash90'

import csv
import numpy as np


def transform(in_file, out_file):
    arr_2d = np.genfromtxt(in_file, delimiter= ",")
    #output = open(out_file,"rw+")
    out_arr = []
    for arr in arr_2d:
        total_load = (arr[0]* arr[1])/25.0 + 1
        if total_load > 1 :
         out_arr.append(total_load)
        else:
         out_arr.append(1)

    np.savetxt(fname = outfile, X = out_arr, fmt = "%.3f" )


infile  = "../datasets/rubis_aws/0107_0108/la"
outfile = "data/la107_108.csv"
transform(infile, outfile)
