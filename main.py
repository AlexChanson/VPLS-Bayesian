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
from skopt import gp_minimize
from skopt.space import Real,Integer
from skopt.utils import use_named_args
import docplex.cp.parameters

##
# Install Ubuntu on your computer for Windows
# Install Python 3.10
# Install CPLEX for Python
# Install Visual Studio Code (Community Version)
##
# To access documentation, use $pydoc3 -p 1234 and connect to localhost:1234/
# ook for the file main.py in your current repertory;
##

params =docplex.cp.parameters.CpoParameters()
#Allow CPLEX to choose the number of threads
params.threads=0
#Makes CPLEX bind each thread to a single core using the available cores
params.cpumask="FFFFFFFF"
#So, CPLEX should use no more than x threads, while x is the number of core of the current machine

##Script configuration
nbCalls=40      #nb of iterations of the bayesian learning (include the number of random iterationsx)
randomIts=10    #nb of random iterations at the beginning of the bayesian learning (included in nbCalls)
maxDelay=10     #nb of minutes allowed for a single cplex call
print("nbCalls="+str(nbCalls))
print("randomIts="+str(randomIts))
print("maxDelay="+str(str(maxDelay)))

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

#
def load_warm_raw(file):
    with open(file) as f:
        S = f.readline().strip().split(" ")
        S = list(map(int, S))
        return S

#
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
    """XOR logical gate
    
    Args:
        i (boolean): first entry
        j (boolean): second entry
    
    Returns:
        a boolean
    """
    if i == 1:
        return 1 - j
    return j

def call_cplex(serialId, size, itTime=3, max_iter=10, h=20, epsTcoef=0.25, epsDcoef=0.35,printing=False):
    """Call CPLEX for particular instance and parameters
    
    This function process a call of the CPLEX solver for a specific instance with a specificsize.

    Args:
        serialId (int): id of the instance to run
        size (int): size of the wanted instance
        itTime (int): maximal time of an iteration in seconds, 3 by default
        max_iter (int): maximal number of tries, 10 by default
        h (int): maximal Hamming distance for solutions, 20 by default
        epsTcoef (float): time limit for the execution time limit constraint of the model, 0.25 by default
        epsDcoef (float): distance limit for the distance limit constraint of the model, 0.35 by default
        printing (boolean): allow to print some logs when True, False by default

    Returns:
        a tuple containing: (the solution,the total resolution time,the objective function value)
    """
    ist_str="./instances/tap_"+str(serialId)+"_"+str(size)
    ist = TapInstance(ist_str+".dat")
    import decimal
    budget = int(decimal.Decimal(str(epsTcoef* ist.size * 27.5)).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))
    dist_bound = int(decimal.Decimal(str(epsDcoef * ist.size * 4.5)).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))
    #Defining the TAP model
    tap = Model(name="TAP")
    tap.set_time_limit(itTime)
    #use only the first core available
    x, s = make_problem(tap, ist, dist_bound, budget)

    previous_solution = load_warm(tap, ist_str+".warm")
    previous_solution.check_as_mip_start(strong_check=True)
    #initiate the time count
    start = time.time()
    current_constraint = None
    for n_iter in range(max_iter):
        tap.add_mip_start(previous_solution.as_mip_start())
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

def find_tap_inst_details(id, size, printing=False):
    """Return interesting information from the file of a particular instance with a certain size
    
    Args:
        id (int): id of the instance
        size (int): size 

    Returns:
        a tuple containing: (time limit, distance limite, resolution time, optimal solution score, solution)
    """
    # open file in read mode
    with open('./data/tap_instances_optimal.csv', 'r') as read_obj:
        #print("File found")
        csv_reader = reader(read_obj)
        next(csv_reader,None)
        for row in csv_reader:
            #Logs for debug if a file is badly writtened
            if printing == True: print("row0="+str(row[0]))
            if printing == True:("row1="+str(row[1]))
            if printing == True:("size="+str(size))
            if printing == True:("id="+str(id))

            #Check if this line is the desired instance
            if int(row[0])==id and int(row[1])==size:
                return (float(row[2]),float(row[3]),float(row[4]),float(row[5]),row[6])
    
    #If not found, returns a tuple fiull of -1 with a log
    print("ERROR: Instance not found!")
    return (-1,-1,-1,-1,"")

def get_instance(id,size):
    """Return datas about a particular instance from its id and size
    
    Args:
        id (int): id of the instance
        size (int): size of the instance

    Returns:
        a tuple containing: (an array[size](float:0-1) with the interest values, an array[size](int) with the runtime values, a matrix[size][size](int) with the distances)
    """
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
    
    #Reformat correctly
    interests=interests.replace("\'","")
    interests=interests.split()
    runtime=runtime.replace("\'","")
    runtime=runtime.split()
    return(interests,runtime,dist)

def relative_error_checker(id, size, z):
    """Compute and return the relative z error in purcentages
    
    Args:
        id (int): id of the instance
        size(int): size of the instance
        z (float): objective function value founded to compare with the best result
        
    Returns:
        a float, the reative difference ((originalZ-zFound)/originalZ) *100
    """
    error_z=0
    #Get the optimal solution
    det = find_tap_inst_details(id,size)
    original_z = det[3]
    #Compute therelative error between current solution and optimal solution
    error_z = ((original_z-z)/original_z)*100.0
    return error_z

def error_difference_checker(id, size, z):
    """Compute and return the z error difference
    
    Args:
        id (int): id of the instance
        size (int): size of the instance
        z (float): objective function value founded to compare with the best result
    """
    error_z=0
    det = find_tap_inst_details(id,size)
    t_z = det[3]
    error_z = t_z-z
    #print("id="+str(id)+";size="+str(size)+";z="+str(z)+";t_z="+str(t_z))
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
                zRelativeError=relative_error_checker(id,size,zDone)
                zError = error_difference_checker(id,size,zDone)
                entries.append((idsPr,id,size,timeDone,zDone,zError,zRelativeError))
                #print("x"+str(idsPr)+";"+str(t_limit)+" "+str(max_iter)+" "+str(h)+";"+str(timeDone)+";"+str(zDone)+";"+str(zError))
                #idsPr+=1
    #print("Processed through IDs "+str(id_all)+" and sizes "+str(size_all))
    zs=[]
    zd=[]
    ze=[]
    timeDoneTot=0
    for row in entries:
        zs.append(row[6])
        zd.append(row[4])
        timeDoneTot+=row[3]
        ze.append(row[5])
    z_error_avg = np.average(zs)#average of z relative error
    z_avg = np.average(zd)#average of z
    z_error=sum(ze)#sum of error
    str_terminal="x"+str(idsPr)+";"+str(t_limit)+";"+str(max_iter)+";"+str(h)+";"+str(timeDoneTot)+";"+str(z_avg)+";"+str(z_error)+";"+str(z_error_avg)
    print(str_terminal)#log for results
    idsPr+=1
    return z_error_avg

#main function
if __name__ == '__main__':
    id=5
    size=200
    t_limit=3
    ids=(0,5)
    sizes=(100,200)
    
    ##Begin bayesian learning
    dim1=Integer(name='tlim', low=60, high=600)
    #dim2=Integer(name='it', low=1, high=50)
    dim3=Integer(name='h', low=1, high=50)
    dims = [dim1,dim3]
    print("xId;t_limit;max_iter;h;timeDone;z;zError;zRelativeError")
    
    @use_named_args(dimensions=dims)
    def prepare(tlim,h):
        it=(maxDelay*60)//tlim
        ist=[]
        #ist.append([(1,2,3,4,5,6,7),(60,80)])
        #ist.append([(1,2,3,4),[100]])
        ist.append([(0,1,2,3,4,5,6,7,8,9),[200]])
        return run_for_ranges(ist, tlim,it, h, 0.25, 0.35)
    
    res = gp_minimize(prepare,                  # the function to minimize
        #[(10,62),(1,50),(1,50)],      # the bounds on each dimension of x
        dimensions=dims,
        acq_func="EI",      # the acquisition function
        n_calls=nbCalls,         # the number of evaluations of f
        n_random_starts=randomIts,  # the number of random initialization points
        random_state=1234)   # the random seed
    print(res)
       
    
    ###
    #####   For further works
    ###

    class histogramer:
        def histoInterest(id,size,nbBins):
            """Compute the histogram of interest for a particular instance with a specified number of bins
            
            Args:
                id (int): id of the instance
                size (int): size of the instance
                nbBins (int): number of bins for the histogram

            Returns:
                a numpy histogram
            """
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
            """Compute the histogram of runtimes for a particular instance with a specified number of bins and an upper bound (recommanded: 11 bins for upper bound at 55)
            
            Args:
                id (int): id of the instance
                size (int): size of the instance
                nbBins (int): number of bins for the histogram
                upper (int): upper bound of values

            Returns:
                a numpy histogram
            """
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
            """Compute the histogram of distances for a particular instance with a specified number of bins and a max distance (recommanded: 11 bins for a max distance of 10)
            
            Args:
                id (int): id of the instance
                size (int): size of the instance
                nbBins (int): number of bins for the histogram
                dmax (int): upper bound of distances

            Returns:
                a numpy histogram
            """
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

    ##Code to warn Alexandre when script is over on the distant server
    try:
        import subprocess
        subprocess.run(["/users/21500078t/discord.sh", "Le script d'Hugo est fini"])
    # execute except block if error occurs
    except ImportError:
        print("Unable to import module.")
    except FileNotFoundError:
        print("File does not exist on this machine.")



    def testall():
        """Call all test functions
        
        Each function shall print as a log the result of the test.

        Args:
            Nothing

        Returns:
            Nothing
        """
        test_errorChecker()
        test_errorDifferenceChecker()
        pass

    def printError(val1,val2,fun=""):
        print("TEST ERROR: "+str(val1)+"=="+str(val2)+" failed in function "+fun)

    def test_errorChecker(id=0,size=20,z=0.06,expected=98.42219446453224):
        res=relative_error_checker(id,size,z)
        if not res == expected:
            printError(res,expected)
            return False
        return True

    def test_errorDifferenceChecker(id=0,size=20,z=0.06,expected=3.74275):
        res=error_difference_checker(id,size,z)
        if not res == expected:
            printError(res,expected)
            return False
        return True

