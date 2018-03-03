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

import subprocess
import pandas

# Use this string to construct the directory where to store the simulation result
dir_res_stor = r'D:\Calibration\amc_init'
variables_sim = 'results.TF[1]'
variables_sim_orig = 'results.TF[1] / degC:#(filter=_Results)'
variables_meas = 'TKF'


# Read the measurements from csv file and export it to pandas data frame
meas = pandas.read_csv(dir_res_stor + r"\meas.csv",";",index_col=0)

# Create the objective function that will simulate the model with the respective
# inputs and return the obejctive function value calculated by the model. 
# The tuple s contains the current values of the decision variables
def objective_op(s):  
    
    #Change to working directory
    os.chdir(dir_res_stor)
    
    #Initial amount of refridgerant in evaporator
   # s1 = s[0]/100*0.8
    #Difference wall - compartment
    #s2 = -5 + s[1]/100*10
    #Difference evaporator - compartment
    #s3 = -10 + s[2]/100*20
    #Compartment temeperature
    s4 = 273 + s[0]/100*18
    
    #Write the dictionary of parameters to calibrate to dsin file
    '''
    modelicares.exps.write_params({'evaporator.initSetUser.mRefRel': s1,
                                   'dT_start_Wall' : s2,
                                   'dTStartEvaporator' : s3,
                                   'cabinet.tStartCompartments[1]' : s4,}, 
                                   dir_res_stor + '\dsin.txt')
    '''
    modelicares.exps.write_params({'cabinet.tStartCompartments[1]' : s4}, 
                                   dir_res_stor + '\dsin.txt')

    #Start a subprocess that simualtes the dymosim.exe
    cmd = dir_res_stor + r'\dymosim.exe'
    subprocess.run(cmd, stdout=subprocess.PIPE)

    # Get the simulation result
    sim = SimRes(dir_res_stor + r'\dsres.mat')
    simf = sim.to_pandas(variables_sim, {variables_sim:variables_sim})
    
    # Define the initial condition for calculating the objective function
    objFun_val = 0

    # At each time sample, calculate the suared error and add it to the 
    # objective function

    #get the time from each row of the measurement frame
    time = np.asscalar(round(meas.index.values[0]))
    #Get the value computed for that time
    sim_value = simf.loc[time, variables_sim_orig]
    #Get the value measured at that time
    meas_value = meas.loc[meas.index.values[0], variables_meas]
    #Calculate the cumulated squared error
    objFun_val = (sim_value - meas_value)**2
    #Store current time for next iteration of the loop

    
    # Calcualte the root of the integral error
    objFun_val = math.sqrt(objFun_val)
    
    #Display this iteration's decision variable values and obejctive function
    #value
    print(s)
    print(objFun_val)
    
    
    #return the negative value for maximization of the error (required for
    # minimization)
    return objFun_val


# Define the initial conditions for the optimization
s0 = [5.0,0.0]

# Define the boundary conditions for the optimization
#b = ((277.15,280.15),(248.15,263.15))
#b = ((4,7),(-25,-10))
b = ((0,100),(0,100))

# Perform a minimization using the defined obejctive function and frequnecies 
# as decision variables
sol = minimize(objective_op,s0,method='SLSQP',bounds=b,
               options={'maxiter':100, 'eps':10, 'ftol':0.0001})

