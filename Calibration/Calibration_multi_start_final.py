# -*- coding: utf-8 -*-
"""
Use this script for model-based testing of controller software 
using Modelica models. The script makes use of the Dymola-Python interface
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
import multiprocessing



# Create the objective function that will simulate the model with the respective
# inputs and return the obejctive function value calculated by the model. 
# The tuple s contains the current values of the decision variables
def objective_op(s,no,ts):  
    
    #Define the local path 
    dir_res_stor_loc = os.path.normpath(r'D:\Calibration_2\amc' + str(no))

    #Define the variables for the simulation and measurements
    variables_sim = {'results.Pel':'results.Pel','results.TF[1]':'results.TF[1]'}
    variables_sim_orig_1 = 'results.Pel / W:#(filter=_Results)'
    variables_meas_1 = 'mess.P'
    variables_sim_orig_2 = 'results.TF[1] / degC:#(filter=_Results)'
    variables_meas_2 = 'TKF'
    
    # Perform a minimization using the defined obejctive function and frequnecies 
    # as decision variables
    
    # Read the measurements from csv file and export it to pandas data frame
    meas = pandas.read_csv(os.path.normpath(dir_res_stor_loc + r"\meas.csv"),";",index_col=0)
    
    #Write the result from the initialisation calibration to the dsin
    s4 = s4 = 273 + ts/100*18
    modelicares.exps.write_params({'cabinet.tStartCompartments[1]' : s4,},
                                  os.path.normpath(dir_res_stor_loc + r'\dsin.txt'))
    
    
    #Change to working directory
    os.chdir(dir_res_stor_loc)
    
    s1 = 4 + s[0]/100*3
    s2 = -25 + s[1]/100*15
    
    #Write the dictionary of parameters to calibrate to dsin file
    modelicares.exps.write_params({'controller.tCompOn': s1, 
                                   'controller.tCompOff': s2,},
                                  os.path.normpath(dir_res_stor_loc + r'\dsin.txt'))

    #Start a subprocess that simualtes the dymosim.exe
    cmd = dir_res_stor_loc + r'\dymosim.exe'
    subprocess.run(cmd, stdout=subprocess.PIPE)

    # Get the simulation result
    sim = SimRes(dir_res_stor_loc + r'\dsres.mat')
    simf = sim.to_pandas(list(variables_sim), variables_sim)
    
    #Search for the beginning of a cycle after the minimum time and drop the 
    #previous lines
    for k in range(700,1200):
        if simf.loc[simf.index[k], variables_sim_orig] > 35:
            break
        
    #simf = simf.drop(simf.index[0:k])

    if k > 1198:
        objFun_val = 50
    else:   
        base_time = simf.index[k]
        
        # Define the initial condition for calculating the objective function
        objFun_val = 0
    
        # At each time sample, calculate the suared error and add it to the 
        # objective function
    
        pre_time = 0
    
        for k in range(0, 1000):
            #get the time from each row of the measurement frame
            time = np.asscalar(round(meas.index.values[k]/10)*10)
            #Get the value computed for that time
            sim_value_1 = simf.loc[time+base_time, variables_sim_orig_1]
            sim_value_2 = simf.loc[time+base_time, variables_sim_orig_2]
            #Get the value measured at that time
            meas_value_1 = meas.loc[meas.index.values[k], variables_meas_1]
            meas_value_2 = meas.loc[meas.index.values[k], variables_meas_2]
            #Calculate the cumulated squared error
            objFun_val = objFun_val + (time-pre_time)*((sim_value_1 - meas_value_1)**2 + 100*(sim_value_2 - meas_value_2)**2 )
            #Store current time for next iteration of the loop
            pre_time = time
        
        #Get start and end time from measurement frame
        startTime = meas.index.values[0]
        totalTime = meas.index.values[len(meas)-1]
        
        # Calcualte the root of the integral error
        objFun_val = math.sqrt(objFun_val/(totalTime-startTime))
    
    #Display this iteration's decision variable values and obejctive function
    #value
    print(s)
    print(ts)
    print(no)
    print(objFun_val)
    
    #return the negative value for maximization of the error (required for
    # minimization)
    return objFun_val


def multiFun(no,ts):

    # Define the initial conditions for the optimization
    s0 = [15,15]
    
    # Define the boundary conditions for the optimization
    #b = ((277.15,280.15),(248.15,263.15))
    #b = ((4,7),(-25,-10))
    b = ((0,100),(0,100))
    
    sol = minimize(objective_op,x0=s0,args=(no,ts),method='SLSQP',bounds=b,
                   options={'maxiter':100, 'eps':5, 'ftol':0.0001})
    
    
    #return sol

if __name__ == '__main__':
    
    global dir_res_str
    dir_res_stor = os.path.normpath(r'D:\01_Dymola_WorkDir\00_Testzone')
    
    ts = range(30,50,2)
    
    for k in range(10):
        test = multiprocessing.Process(target=multiFun, args=(k,ts[k]))
        test.start()
        #test.join()
        
    
    #sol = multiOptim(no)
    

