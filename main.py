import sys
import numpy as np
import time

from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from docplex.mp.model import *
from docplex.mp.solution import SolveSolution
from structures import TapInstance

#lib Hugo
from csv import reader

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

'''
def call_cplex_safemode():
    ist = TapInstance("./instances/tap_5_200.dat")
    max_iter = 10
    h = 20
    #.solv -> rajouter t_it max (voir doc)
    #return solution, t_exec...
    budget = round(0.25 * ist.size * 27.5)#0,25 -> param e_t
    dist_bound = round(0.35 * ist.size * 4.5)#0,35 -> param e_d
    print(ist)

    tap = Model(name="TAP")
    tap.set_time_limit(3)
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
    return (previous_solution,tap.solve_details.time,previous_solution.get_objective_value())
'''

def call_cplex(serialId, size, itTime=5, max_iter=10, h=20, epsTcoef=0.25, epsDcoef=0.35):
    #sId=5, size=200
    ist_str="./instances/tap_"+str(serialId)+"_"+str(size)
    ist = TapInstance(ist_str+".dat")
    #max_iter = 10
    #h = 20
    #.solv -> rajouter t_it max (voir doc)
    #return solution, t_exec...
    budget = round(epsTcoef * ist.size * 27.5)#0,25 -> param e_t
    dist_bound = round(epsDcoef * ist.size * 4.5)#0,35 -> param e_d
    print(ist)
    tap = Model(name="TAP")
    tap.set_time_limit(itTime)
    x, s = make_problem(tap, ist, dist_bound, budget)
    #tap.print_information()
    #solution = tap.solve()
    #tap.print_solution()

    previous_solution = load_warm(tap, ist_str+".warm")
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
    return (previous_solution,tap.solve_details.time,previous_solution.get_objective_value())
    pass

def find_tap_inst_details(id, size):
    # open file in read mode
    with open('./data/tap_instances_optimal.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            if row[0]==id and row[1]==size:
                return (row[2],row[3],row[4],row[5],row[6])
        pass
    return (-1,-1,-1,-1,"")

def error_checker(id, size, z, sol=""):
    error_z=0
    error_sol=0
    det = find_tap_inst_details(id,size)
    t_z = det[3]
    t_s = det[4]
    error_z = abs(z-t_z)
    c_sol=[]
    t_sol = []
    ts_first = True
    sfirst = True
    i=0
    '''
    for c in sol:
        if c == ',':
            sfirst=True
        elif not c=="\"":
            if sfirst:
                sfirst=False
                c_sol.append(int(c))
            else:
                c_sol[i] *=10
                c_sol[i] +=int(c)
        
        if t_s[i] == ',':
            ts_first=True
        elif not t_s[i]=="\"":
            if ts_first:
                ts_first=False
                t_sol.append(int(c))
            else:
                t_sol[i] *=10
                t_sol[i] +=int(c)
    error_sol = hammingDistance(c_sol,t_sol)
    return (error_z,error_sol)
    '''
    return error_z

def hammingDistance(v1,v2):
    h=0
    for i in range(v1.length()):
        if not v1[i] == v2[i]:
            h+=1
    return h

def run_for_ranges(id_all, size_all, t_limit):
    for size in size_all:
        for id in id_all:
            solv=call_cplex(id,size,t_limit)
            #(SolveSolution) solv[0].get_var_value()
            timeDone = solv[1]
            zDone = solv[2]
            zError=error_checker(id,size,zDone)
            print("===\nTime done: "+str(timeDone)+"\nz: "+str(zDone)+"\nz error: "+str(zError)+"\n")
    print("Processed through IDs "+str(id_all)+" and sizes "+str(size_all))

if __name__ == '__main__':
    id=5
    size=200
    t_limit=3
    run_for_ranges((0,5),(100,200),t_limit)
    #solv=call_cplex(id,size,t_limit)
    #print("Solution")
    #for e in solv[0]:
    #    print(str(e))
    #print("time::"+str(solv[1])+"::z::"+str(solv[2]))
    #err = error_checker(id,size,solv[0],solv[2])
    #print("Erreur::"+str(err[0])+"::"+str(err[1]))
