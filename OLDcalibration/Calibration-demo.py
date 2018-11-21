# -*- coding: utf-8 -*-
"""
Use this script for model-based testing of controller software 
using Modelica models. The script makes use of the Dymola-Python interface
for automatizing simulations. It uses modelicares for reading and analyzing
simulation 
It further uses numpy for generating signals and a non-linear minimization 
algorithm from scipy .
"""

import os
import sys
import numpy as np
# import modelicares
from modelicares import SimRes
from scipy.optimize import minimize
from scipy import math
import matplotlib as mpl
mpl.use('QT5Agg')
import matplotlib.pyplot as plt
import time

# import scipy.io as sio
# from scipy import signal

# Import Dymola Package. How To: https://github.com/RWTH-EBC/AixLib/wiki/How-to:-Dymola-Python-Interface
from dymola.dymola_interface import DymolaInterface

# Start the interface
dymola = DymolaInterface()

# AixLib directory
dir_Aix = os.path.join(r'D:\04_Git\AixLib_development\AixLib\AixLib\package.mo')

# Use this string to construct the directory where to store the simulation result
dir_res = r'D:\01_Dymola_WorkDir\00_Testzone\pyInterface'

dir_res_stor = dir_res + r'\dsres'

dir_res_open = dir_res_stor + r'.mat'

dir_res_stor_ref = dir_res + r'\dsresRef'

dir_res_open_ref = dir_res_stor_ref + r'.mat'

# Use this path to access the model to be used in this study
path = 'AixLib.Fluid.HeatPumps.Examples.HeatPumpDetailed'

# Simulation time
totalTime = 20000
t = np.zeros(totalTime + 1)

# Create an array containing the time stamp
for k in range(0, totalTime + 1):
    t[k] = float(k)

# Name of the controlled variable
variables_of_interest = ['heatPump.T_conOut.T', 'heatPump.P_eleOut']
name_cv1 = variables_of_interest[0]
name_cv2 = variables_of_interest[1]

# Do not store at events
dymola.experimentSetupOutput(events=False)

# Change working directory for python and dymola instance
os.chdir(dir_res)
dymola.cd(dir_res)
# Open AixLib and store the returned Boolean indicating successful opening
check1 = dymola.openModel(dir_Aix, changeDirectory=False)
# Translate the model
tic = time.time()
check2 = dymola.translateModel(path)
print('\nTime for model translation lasted : ' + str(np.round(time.time() - tic, 2)) + ' s')
if not check1 and not check2:
    raise ValueError('\nLoading library and / or Translation was not successful')


def objective(s):
    # Call the simulation function
    try:
        values_res = simulation(names, s.tolist(), dir_res_stor, dir_res_open)

        print('Current values are: \n' + str(s))

        objFun_val = 0
        objFun_val1 = 0
        objFun_val2 = 0
        sd1 = 0
        sd2 = 0

        # At each time sample, calculate the squared error and add it to the 
        # objective function

        mean1 = np.sum(values_ref[:, 0]) / np.size(values_ref[:, 0])

        mean2 = np.sum(values_ref[:, 1]) / np.size(values_ref[:, 1])

        for k in range(0, np.size(values_ref[:, 0])):
            sd1 = sd1 + ((mean1 - values_ref[k, 0]) ** 2)

        for k in range(0, np.size(values_ref[:, 1])):
            sd2 = sd2 + ((mean2 - values_ref[:, 1]) ** 2)

        for k in range(0, np.size(values_ref[:, 0])):
            objFun_val1 = objFun_val1 + (values_ref[k, 0] - values_res[k, 0]) ** 2
            objFun_val2 = objFun_val2 + (values_ref[k, 1] - values_res[k, 1]) ** 2

        # objFun_val1 = math.sqrt(objFun_val1)/math.sqrt(sd1)
        # objFun_val2 = math.sqrt(objFun_val2)/math.sqrt(sd2)

        # Sum of all single objective functions
        objFun_val = objFun_val1 + objFun_val2

        print('Value of the obejctive function is: \n' + str(objFun_val))

        # Calcualte the root of the integral error
        # objFun_val = math.sqrt(objFun_val/(totalTime-startTime))

    except:
        raise SystemError('Creation of objective function does not work.')

    return objFun_val


Nfeval = 1
fig_optimizer, ax_optimizer = plt.subplots(nrows=1, ncols=1)
plt.show(block=False)
print('Ueber figure erstellung hinweg')
def callbackF(Xi):
    global Nfeval
    value_obj_func = objective(Xi)
    print('Number of iteration: {}, value objective function: {}'.format(Nfeval, value_obj_func))
    ax_optimizer.plot(Nfeval, value_obj_func, color='b',marker='x')
    # Figure will be totally visible if breakpoint in line hereafter
    Nfeval += 1

def simulation(names, s, direc1, direc2):
    if not isinstance(names, list) or not isinstance(s, list):
        raise AttributeError('names and s must be of type list')
    # Run the simulation with the given initial values and output increment
    dymola.simulateExtendedModel(
        problem=path,
        startTime=0.0,
        stopTime=totalTime,
        outputInterval=1,
        method="Dassl",
        tolerance=0.0001,
        resultFile=direc1,
        initialNames=names,
        initialValues=s
    )

    # Get the simulation result
    sim = SimRes(direc2)

    # Extract trajectory
    # AS PANDAS DATA FRAME: values_res = sim.to_pandas(variables_of_interest)
    values_res = np.column_stack((sim[name_cv1].values(), sim[name_cv2].values()))

    return values_res


# Define names and according values for simulation parameters
names = ["heatPump.dataTable.tableQdot_con[2, 2]", "heatPump.dataTable.tableQdot_con[2, 3]"]
            #,"heatPump.dataTable.tableQdot_con[3, 2]", "heatPump.dataTable.tableQdot_con[3, 3]"]
s = [4800, 6300]#, 4400, 5750]
if not len(names)==len(s):
    raise ValueError('Variable {} and {} must have same length'.format('names', 'values'))
# Generate reference trajectory
values_ref = simulation(names, s, dir_res_stor_ref, dir_res_open_ref)

# Define the initial conditions for the optimization
s0 = [4700, 6000]#[2000, 2000, 2000, 2000]

# Define the boundary conditions for the optimization
b = ((4650, 4850), (5900, 6500))#((500, 10000), (500, 10000), (500, 10000), (500, 10000))

# Perform a minimization using the defined obejctive function and frequnecies 
# as decision variables
iter_no = 0 # number of iterations
sol = minimize(fun=objective, x0=s0, method='SLSQP', bounds=b, options={'maxiter': 1000, 'eps': 200, 'ftol': 0.01}, callback=callbackF)

values_final = simulation(names, sol.x.tolist(), dir_res_stor, dir_res_open)

if dymola is not None:
    dymola.close()
    dymola = None
