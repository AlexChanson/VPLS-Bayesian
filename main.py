import sys
import cplex
from cplex.exceptions import CplexError
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern


def load_instance(file):
    with open(file) as f:
        n = int(f.readline())
        I = []
        for e in f.readline().split(" "):
            I.append(float(e))
        T = []
        for e in f.readline().split(" "):
            T.append(float(e))
        D = []
        for i in range(n):
            d = []
            for e in f.readline().split(" "):
                d.append(float(e))
            D.append(d)
    return n, I, T, D


if __name__ == '__main__':
    tap = cplex.Cplex()
    print(tap)
    ist = load_instance("/home/alex/instances/tap_1_20.dat")
    print(ist)
