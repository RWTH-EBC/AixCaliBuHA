# -*- coding: utf-8 -*-
"""
Use this script for model-based testing of controller software 
unsing Modelica models. The script makes use of the Dymola-Python interface
for automatizing simulations. It uses modelicares for reading and analyzing
simulation results and also for design of experiment (DOE) functions.
It further uses numpy for generating signals and a non-linear minimization 
algorithm from scipy .
"""

import os
import sys
import numpy as np
import modelicares
from modelicares import SimRes
from scipy.optimize import minimize
from scipy import math


import scipy.io as sio
from scipy import signal
import subprocess

# Work-around for the environment variable
sys.path.insert(0, os.path.join('C:\\',
                                'Program Files (x86)',
                                'Dymola 2018',
                                'Modelica',
                                'Library',
                                'python_interface',
                                'dymola.egg'))


# Import Dymola Package
from dymola.dymola_interface import DymolaInterface

# Start the interface
#dymola = DymolaInterface()

base_path = r'D:\Git\AixLib\AixLib\Airflow\FacadeVentilationUnit\Examples'

# Use this path to access the model to be used in this study
path = r'AixLib.Airflow.FacadeVentilationUnit.Examples.FacadeVentilationUnitInput'

# AixLib directory
dir_Aix = os.path.join(r'D:\Git\AixLib\AixLib\package.mo')

# Use this string to construct the directory where to store the simulation result
dir_res_stor = base_path + r'\dsres'

# Use this path to open the simulation results
dir_res_open = base_path + r'\dsres.mat'

os.chdir(base_path)

# Open AixLib and store the returned Boolean indicating successful opening
#check1 = dymola.openModel(dir_Aix)

# Translate the model
#check2 = dymola.translateModel(path)

# Simulation time
totalTime=86400

# Create an array containing the time stamp
t = np.linspace(0, totalTime, totalTime)
t = t.reshape(1,totalTime)

def run(*popenargs, input=None, check=False, **kwargs):
    if input is not None:
        if 'stdin' in kwargs:
            raise ValueError('stdin and input arguments may not both be used.')
        kwargs['stdin'] = subprocess.PIPE

    process = subprocess.Popen(*popenargs, **kwargs)
    try:
        stdout, stderr = process.communicate(input)
    except:
        process.kill()
        process.wait()
        raise
    retcode = process.poll()
    if check and retcode:
        raise subprocess.CalledProcessError(
            retcode, process.args, output=stdout, stderr=stderr)
    return retcode, stdout, stderr

# Create the objective function that will simulate the model with the respective
# inputs and return the obejctive function value calculated by the model. 
# The tuple s contains the current values of the decision variables
def objective(s):  
    
    # Use this vector to create the first input signal. The chirp signal 
    # becomes a sine wave, whereas the frequnecy is a decision variable
    vect_1 = (273.15+10*(signal.chirp(t,s[0], 1, s[0]).reshape((1,totalTime))))

    # Use this array to store a .mat file that can be read by Dymola. This 
    # file inputs the trajectory of the ambient temperature
    arr_1 = np.append(t, vect_1, axis=0)
    arr_trans = np.transpose(arr_1)
    sio.savemat('externalAmbient.mat', {'ambient':arr_trans})
    
    # Use this vector to create the second input signal. The chirp signal 
    # becomes a sine wave, whereas the frequnecy is a decision variable
    vect_2 = (200*(signal.chirp(t,s[1], 1, s[1]).reshape((1,totalTime))))

    # Use this array to store a .mat file that can be read by Dymola. This 
    # file inputs the trajectory of the flow temperature
    arr_2 = np.append(t, vect_2, axis=0)
    arr_trans_2 = np.transpose(arr_2)
    sio.savemat('disturbance.mat', {'disturbance':arr_trans_2})
    
    # Name of the controlled variable
    name_cv = 'roomTemperatureMeasurement.T'
    
    # Name of the set point variable
    name_sp = 'roomSetTemperature.y'
    
    # Time increment
    incr = 10
    
    # Simulate the model using the total time and increment defined before. 
    # Store the results in the specified result file.
    
    cmd = base_path + r'\dymosim.exe'
    run(cmd, stdout=subprocess.PIPE)    
    '''
    dymola.simulateExtendedModel(
    problem=path,
    startTime=0.0,
    stopTime=totalTime,
    outputInterval=incr,
    method="Dassl",
    tolerance=0.0001,
    resultFile=dir_res_stor,
    finalNames = [name_cv,name_sp]
    )
    '''
    # Get the simulation result
    sim = SimRes(dir_res_open)

    # Extract the trajectory of the controlled variable
    values_cv = sim[name_cv].values() 
    
    # Extract the trajectory of the set point variable
    values_sp = sim[name_sp].values() 
    
    # Define the initial condition for calculating the objective function
    objFun_val = 0
    startTime = 1000

    # At each time sample, calculate the suared error and add it to the 
    # objective function
    for k in range(int(startTime/incr),int(totalTime/incr)):
        objFun_val = objFun_val + incr*(values_cv[k]-values_sp[1])**2
    
    # Calcualte the root of the integral error
    objFun_val = math.sqrt(objFun_val/(totalTime-startTime)*incr)
    
    #return the negative value for maximization of the error (required for
    # minimization)
    return -objFun_val


# np.linspace(0, totalTime, totalTime)

# Use the fullfactorial design of experiment: define the upper and lower bounds
# and the increment
x_min = 1/86400
x_max = 1/1200
x_incr = 1/3600
y_min = 1/86400
y_max = 1/1200
y_incre = 1/3600

settings = modelicares.exps.doe.fullfact([x_max,x_incr,x_min],[y_max,y_incre,y_min])
counter = 0

storage = np.empty([9,3])

for s in settings:
    storage[counter,0] = 1/s[0]
    storage[counter,1] = 1/s[1]
    storage[counter, 2] = objective(s)
    counter = counter + 1

# Define the initial conditions for the optimization
s0 = [1/86400,1/86400]

# Define the boundary conditions for the optimization
b = ((1/200000,1/1200),(1/200000,1/1200))

# Perform a minimization using the defined obejctive function and frequnecies 
# as decision variables
sol = minimize(objective,s0,method='SLSQP',bounds=b,options={'maxiter':100, 'eps':0.001, 'ftol':0.0001})

# Store the minimized value
minValue = objective(sol.x)
     
'''       
# Exit Dymola
if dymola is not None:
    dymola.close()
    dymola = None
'''

