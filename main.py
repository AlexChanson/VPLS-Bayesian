import sys

import cplex
from cplex.exceptions import CplexError

if __name__ == '__main__':
    my_prob = cplex.Cplex()
    print(my_prob)
