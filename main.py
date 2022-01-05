import sys
import numpy as np
import time

from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from docplex.mp.model import *
from docplex.mp.solution import SolveSolution
from structures import TapInstance


def make_problem(prob, instance, ed, et):
    n = instance.size
    s = prob.binary_var_list([i for i in range(n)], 0, 1, ["s_" + str(i+1) for i in range(n)])
    x = prob.binary_var_matrix([i for i in range(n + 2)], [i for i in range(n + 2)], name=lambda i: "x_"+str(i[0])+"_"+str(i[1]))
    u = prob.integer_var_list([i for i in range(n)], 2, n, ["u_" + str(i+1) for i in range(n)])

    # Objective : for each query if it's there count it's interest
    prob.set_objective("max", sum([s[i] * instance.interest[i] for i in range(n)]))

    # Distance bound
    prob.add_constraint(sum([x[i, j] * instance.dist[i-1][j-1] for j in range(1, n+1) for i in range(1, n+1)]) <= ed, ctname="distance_epsilon")

    # Time bound
    prob.add_constraint(sum([s[i] * instance.time[i] for i in range(n)]) <= et, ctname="time_epsilon")

    # inbound = outbound = s
    for j in range(1, n + 1):
        prob.add_constraint(sum([x[i, j] for i in range(n + 1)]) == s[j - 1], ctname='inbound' + str(j))

    for i in range(1, n + 1):
        prob.add_constraint(sum([x[i, j] for j in range(1, n + 2)]) == s[i - 1], ctname='outbound' + str(i))

    # only one arc leaves start only on enters end
    prob.add_constraint(sum([x[0, i] for i in range(1, n + 1)]) == 1, ctname="path_start")
    prob.add_constraint(sum([x[i, n + 1] for i in range(1, n + 1)]) == 1, ctname="path_end")

    # no self loops , no start->end
    prob.add_constraint(x[0, n + 1] == 0, ctname="path_structure_0")
    prob.add_constraint(sum([x[i, i] for i in range(0, n+2)]) == 0, ctname="path_structure_1")

    # MTZ subtour elimination
    for i in range(1, n+1):
        for j in range(1, n+1):
            prob.add_constraint(((n - 1) * (1 - x[i, j])) - u[i - 1] + u[j - 1] >= 1, ctname="mtz_" + str(i) + "," + str(j))
    return x, s


def load_warm_raw(file):
    with open(file) as f:
        S = f.readline().strip().split(" ")
        S = list(map(int, S))
        return S


def load_warm(prob, file) -> SolveSolution:
    # Load base solution
    warm = load_warm_raw(file)
    start = SolveSolution(prob)
    for s in warm:
        start.add_var_value("s_" + str(s + 1), 1)
    for i in range(len(warm) - 1):
        this = warm[i] + 1
        next_ = warm[i + 1] + 1
        start.add_var_value("x_" + str(this) + "_" + str(next_), 1)
    check = start.check_as_mip_start()
    return start


def vpls_xor(i, j):
    if i == 1:
        return 1 - j
    return j


if __name__ == '__main__':
    ist = TapInstance("./instances/tap_5_200.dat")
    max_iter = 10
    h = 20
    budget = round(0.25 * ist.size * 27.5)
    dist_bound = round(0.35 * ist.size * 4.5)
    print(ist)

    tap = Model(name="TAP")
    x, s = make_problem(tap, ist, dist_bound, budget)

    #tap.print_information()
    #solution = tap.solve()
    #tap.print_solution()

    previous_solution = load_warm(tap, "./instances/tap_5_200.warm")
    previous_solution.check_as_mip_start(strong_check=True)
    start = time.time()
    current_constraint = None
    for n_iter in range(max_iter):
        tap.add_mip_start(previous_solution.as_mip_start())
        #print([int(previous_solution.get_var_value(s[i])) for i in range(ist.size)])
        print(previous_solution.get_objective_value())
        if current_constraint is not None:
            tap.remove_constraint(current_constraint)
        s_vals = [int(previous_solution.get_var_value(s[i])) for i in range(ist.size)]

        tap.add_constraint(sum([vpls_xor(s_vals[i], s[i]) for i in range(ist.size)]) <= h, ctname="vpls")
        current_constraint = tap.get_constraint_by_name("vpls")

        previous_solution = tap.solve()
        print(tap.get_solve_status())

    end = time.time()
    print("Time (s):", end - start)
