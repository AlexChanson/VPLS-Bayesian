from pickletools import optimize
import re
import sys
import matplotlib
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
import sklearn as sk
#from sklearn import linear_model as lin_mod
from skopt import gp_minimize
from skopt.space import Real,Integer
from skopt.utils import use_named_args
from matplotlib import pyplot as plt

import docplex.cp.parameters
params =docplex.cp.parameters.CpoParameters()
params.threads=0
params.cpumask="FFFFFFFF"

#Script config
nbCalls=40
randomIts=10
print("nbCalls="+str(nbCalls))
print("randomIts="+str(randomIts))

##Define the mathematical problem of the problem we want to solve
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

def moy(liste):
    res=0
    i=0
    for item in liste:
        res+=item
        i+=1
    res/=i
    return res

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
        start.add_var_value("x_" + str(this) + "_" + str(next_), 
        1)
    check = start.check_as_mip_start()
    return start


def vpls_xor(i, j):
    if i == 1:
        return 1 - j
    return j

##Call CPLEX for particular instance and parameters
def call_cplex(serialId, size, itTime=3, max_iter=10, h=20, epsTcoef=0.25, epsDcoef=0.35,printing=False):
    #sId=5, size=200
    ist_str="./instances/tap_"+str(serialId)+"_"+str(size)
    ist = TapInstance(ist_str+".dat")
    #max_iter = 10
    #h = 20
    #.solv -> rajouter t_it max (voir doc)
    #return solution, t_exec...
    import decimal
    budget = int(decimal.Decimal(str(epsTcoef* ist.size * 27.5)).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))
    dist_bound = int(decimal.Decimal(str(epsDcoef * ist.size * 4.5)).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))
    #0,25 -> param e_t
    #0,35 -> param e_d
    #print(ist)
    tap = Model(name="TAP")
    tap.set_time_limit(itTime)
    #use only the first core available
    x, s = make_problem(tap, ist, dist_bound, budget)

    previous_solution = load_warm(tap, ist_str+".warm")
    previous_solution.check_as_mip_start(strong_check=True)
    start = time.time()
    current_constraint = None
    for n_iter in range(max_iter):
        tap.add_mip_start(previous_solution.as_mip_start())
        #if printing==True:print([int(previous_solution.get_var_value(s[i])) for i in range(ist.size)])
        if printing==True:print(previous_solution.get_objective_value())
        if current_constraint is not None:
            tap.remove_constraint(current_constraint)
        s_vals = [int(previous_solution.get_var_value(s[i])) for i in range(ist.size)]

        tap.add_constraint(sum([vpls_xor(s_vals[i], s[i]) for i in range(ist.size)]) <= h, ctname="vpls")
        current_constraint = tap.get_constraint_by_name("vpls")

        previous_solution = tap.solve()
        if printing==True:print(tap.get_solve_status())
    end = time.time()
    if printing==True:print("Time (s):", end - start)
    return (previous_solution,tap.solve_details.time,previous_solution.get_objective_value())

##Return interesting information from the file of a particular instance with a certain size
def find_tap_inst_details(id, size):
    # open file in read mode
    with open('./data/tap_instances_optimal_fixed.csv', 'r') as read_obj:
        #print("ok")
        csv_reader = reader(read_obj)
        next(csv_reader,None)
        for row in csv_reader:
            #print("row0="+str(row[0]))
            #print("row1="+str(row[1]))
            #print("size="+str(size))
            #print("id="+str(id))
            if int(row[0])==id and int(row[1])==size:
                return (float(row[2]),float(row[3]),float(row[4]),float(row[5]),row[6])
    print("ERROR: Instance not found!")
    return (-1,-1,-1,-1,"")

def get_instance(id,size):
    # open file in read mode
    file_str="./instances/tap_"+str(id)+"_"+str(size)+".dat"
    f=open(file_str, 'r')
    lines = f.readlines()
    
    c=0
    interests=""
    runtime=""
    dist=[]
    for line in lines:
        line=line.strip()
        if c==1:
            interests=line
        elif c==2:
            runtime=line
        elif c>=3:
            dist.append(line)
        c+=1
    interests=interests.replace("\'","")
    interests=interests.split()
    runtime=runtime.replace("\'","")
    runtime=runtime.split()
    return(interests,runtime,dist)


##Compute and return the relative z error in purcentages
def error_checker(id, size, z):
    error_z=0
    det = find_tap_inst_details(id,size)
    t_z = det[3]
    error_z = ((t_z-z)/t_z)*100.0
    print("id="+str(id)+";size="+str(size)+";z="+str(z)+";t_z="+str(t_z))
    return error_z

idsPr=1
##Call CPLEX several time on a range of instance IDs and a range of sizes
def run_for_ranges(instances, t_limit,max_iter, h, epsTcoef=0.25, epsDcoef=0.35):
    global idsPr
    #print(str(t_limit)+"###"+str(max_iter)+"##"+str(h))
    entries=[]
    for inst in instances:
        for size in inst[1]:
            for id in inst[0]:
                solv=call_cplex(id,size,t_limit,max_iter, h, epsTcoef, epsDcoef)
                timeDone = solv[1]
                zDone = solv[2]
                zError=error_checker(id,size,zDone)
                entries.append((idsPr,id,size,timeDone,zDone,zError))
                #print("x"+str(idsPr)+";"+str(t_limit)+" "+str(max_iter)+" "+str(h)+";"+str(timeDone)+";"+str(zDone)+";"+str(zError))
                #idsPr+=1
    #print("Processed through IDs "+str(id_all)+" and sizes "+str(size_all))
    zs=[]
    zd=[]
    timeDoneTot=0
    for row in entries:
        zs.append(row[5])
        zd.append(row[4])
        timeDoneTot+=row[3]
    z_error_avg = moy(zs)
    z_avg = moy(zd)
    str_terminal="x"+str(idsPr)+";"+str(t_limit)+";"+str(max_iter)+";"+str(h)+";"+str(timeDoneTot)+";"+str(z_avg)+";"+str(z_error_avg)
    print(str_terminal)
    idsPr+=1
    return z_error_avg

#main function
if __name__ == '__main__':
    id=5
    size=200
    t_limit=3
    ids=(0,5)
    sizes=(100,200)
    '''
    raw=run_for_ranges((0,5),(100,200),t_limit)
    
    zs=[]
    for row in raw:
        zs.append(row[4])
    #z_error_avg = moy(zs)
    z_error_avg=raw
    print("Error average: "+str(z_error_avg))
    #print("avg z="+str(np.average(raw[4])))
    #solv=call_cplex(id,size,t_limit)
    #print("time::"+str(solv[1])+"::z::"+str(solv[2]))
    #err = error_checker(id,size,solv[0],solv[2])
    #print("Erreur::"+str(err[0])+"::"+str(err[1]))
    '''
    ##begin bayes
    dim1=Integer(name='tlim', low=60, high=600)
    dim2=Integer(name='it', low=1, high=50)
    dim3=Integer(name='h', low=1, high=50)
    dims = [dim1,dim2,dim3]
    print("xId;t_limit;max_iter;h;timeDone;z;zRelativeError")
    
    @use_named_args(dimensions=dims)
    def prepare(tlim,it,h):
        ist=[]
        ist.append([(1,2,3,4,5,6,7),(60,80)])
        ist.append([(1,2,3,4),[100]])
        ist.append([(1,2),[200]])
        return run_for_ranges(ist, tlim,it, h, 0.25, 0.35)
    res = gp_minimize(prepare,                  # the function to minimize
        #[(10,62),(1,50),(1,50)],      # the bounds on each dimension of x
        dimensions=dims,
        acq_func="EI",      # the acquisition function
        n_calls=nbCalls,         # the number of evaluations of f
        n_random_starts=randomIts,  # the number of random initialization points
        random_state=1234)   # the random seed
    print(res)
    #from skopt.plots import plot_convergence
    #convergence=plot_convergence(res)
    #plt.plot(convergence)
    #plt.show()


    #histV=np.histogram()
    histC=0
    histT=0

    def histoInterest(id,size,nbBins):
        bins=[]
        beans=[]
        for i in range(0,nbBins):
            beans.append(0)
            bins.append((1.0/nbBins)*(i+1.0))
        #get interests
        inst=get_instance(id,size)
        interests=inst[0]
        for k in interests:
            machined=False
            j=0
            i=float(k)
            if i<0:
                machined=True
                print("ERROR: Negative interest value!")
            while not machined or j<nbBins:
                if i<bins[j]:
                    beans[j]+=1
                    machined=True
                j+=1
        total=np.sum(beans)
        freq=[]
        for b in beans:
            freq.append(int(b)/total)
        return (bins,freq)

    def histoRuntime(id,size,nbBins=11,upper=50):
        bins=[]
        beans=[]
        for i in range(0,nbBins):
            beans.append(0)
            bins.append((upper/int(nbBins))*(i+1))
        beans.append(0)
        bins.append(999)
        #get runtimes
        inst=get_instance(id,size)
        runtime=inst[1]
        for k in runtime:
            machined=False
            j=0
            r=float(k)
            if r<0:
                machined=True
                print("ERROR: Negative interest value!")
            while not machined or j<nbBins:
                if r<bins[j]:
                    beans[j]+=1
                    machined=True
                j+=1
        total=np.sum(beans)
        freq=[]
        for b in beans:
            freq.append(int(b)/total)
        return (bins,freq)

    def histoDistances(id,size,nbBins=11,dmax=10):
        bins=[]
        beans=[]
        for i in range(0,nbBins):
            beans.append(0)
            bins.append((dmax/nbBins)*(i+1))
        beans.append(0)
        bins.append(999)
        #get distances
        inst=get_instance(id,size)
        dists=inst[2]
        distances=[]
        for d in dists:
            x=d.split()
            matrixRight=False
            for y in x:
                if(matrixRight):
                    distances.append(int(y))
                else:
                    if int(y)==0:
                        matrixRight=True
        for k in distances:
            machined=False
            j=0
            d=float(k)
            if d<0:
                machined=True
                print("ERROR: Negative interest value!")
            while not machined or j<nbBins:
                if d<bins[j]:
                    beans[j]+=1
                    machined=True
                j+=1
        total=np.sum(beans)
        freq=[]
        for b in beans:
            freq.append(int(b)/total)
        return (bins,freq)

    '''
    hI=histoInterest(0,20,10)
    print(np.sum(hI[1]))
    print(hI[0])
    hT=histoRuntime(0,20,10)
    print(np.sum(hT[1]))
    print(hT[0])
    hD=histoDistances(0,20,10)
    print(np.sum(hD[1]))
    print(hD[0])

    ci=1
    dimensions=[]
    for u in hI[1] :
        dimensions.append(Integer(name="a"+str(ci), low=60, high=600))
        ci+=1
    for u in hT[1] :
        dimensions.append(Integer(name="a"+str(ci), low=60, high=600))
        ci+=1
    for u in hD[1] :
        dimensions.append(Integer(name="a"+str(ci), low=60, high=600))
        ci+=1
    for u in range(0,3) :
        dimensions.append(Integer(name="a"+str(ci), low=60, high=600))
        ci+=1
    print(ci)
    

    def linearComb():
        summ=0
    '''

    try:
        import subprocess
        subprocess.run(["/users/21500078t/discord.sh", "Le script d'Hugo est fini"])
    # execute except block if error occurs
    except ImportError:
        print("Unable to import module.")
    except FileNotFoundError:
        print("File does not exist on this machine.")
