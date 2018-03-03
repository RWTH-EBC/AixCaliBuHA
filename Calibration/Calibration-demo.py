# -*- coding: utf-8 -*-
"""
Use this script for model-based testing of controller software 
unsing Modelica models. The script makes use of the Dymola-Python interface
for automatizing simulations. It uses modelicares for reading and analyzing
simulation 
It further uses numpy for generating signals and a non-linear minimization 
algorithm from scipy .
"""

import os
import sys
import numpy as np
#import modelicares
from modelicares import SimRes
from scipy.optimize import minimize
from scipy import math


#import scipy.io as sio
#from scipy import signal

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
dymola = DymolaInterface()

# AixLib directory
dir_Aix = os.path.join('D:\Git\AixLib\AixLib\package.mo')

# Use this string to construct the directory where to store the simulation result
dir_res_stor = 'D:\TEMP\dsres'

# Use this path to open the simulation results
dir_res_open = 'D:\TEMP\dsres.mat'

dir_res_open_ref = 'D:\TEMP\dsresRef.mat'

dir_res_stor_ref = 'D:\TEMP\dsresRef'

# Use this path to access the model to be used in this study
path = 'AixLib.Fluid.HeatPumps.Examples.HeatPumpDetailedLarge'

#os.chdir('D:\TEMP')

# Open AixLib and store the returned Boolean indicating successful opening
check1 = dymola.openModel(dir_Aix)

# Translate the model
check2 = dymola.translateModel(path)

# Simulation time
totalTime=20000
t=np.zeros(totalTime+1)
incr = 1

# Create an array containing the time stamp
for k in range(0,totalTime+1):
    t[k] = float(k)   
    
# Name of the controlled variable
name_cv1 = 'temperature.T'
name_cv2 = 'temperature1.T'

# Do not store at events
dymola.experimentSetupOutput(events=False)


def objective(s):    

    #sim = SimRes(dir_res_open_ref)
    
    # Extract trajectory 
    #values_ref1 = sim[name_cv1].values() 
    #values_ref2 = sim[name_cv2].values() 
    
    # Call the simulation function
    try:
        values_res = simulation([s[0],s[1],s[2],s[3],s[4]], dir_res_stor, dir_res_open)
        
        #values_res = values_ref
        print(s)
             
        objFun_val = 0
        objFun_val1 = 0
        objFun_val2 = 0
        sd1 = 0
        sd2 = 0
    
        # At each time sample, calculate the squared error and add it to the 
        # objective function

        mean1 = np.sum(values_ref[:,0])/np.size(values_ref[:,0])

        mean2 = np.sum(values_ref[:,1])/np.size(values_ref[:,1])
            
        for k in range(0,np.size(values_ref[:,0])):
            sd1 = sd1 + ((mean1-values_ref[k,0])**2)
            
            
        for k in range(0,np.size(values_ref[:,1])):
            sd2 = sd2 + ((mean2-values_ref[:,1])**2)
            
        
        for k in range(0,np.size(values_ref[:,0])):
            objFun_val1 = objFun_val1 + (values_ref[k,0]-values_res[k,0])**2 
            objFun_val2 = objFun_val2 + (values_ref[k,1]-values_res[k,1])**2
        
        
        #objFun_val1 = math.sqrt(objFun_val1)/math.sqrt(sd1)
        #objFun_val2 = math.sqrt(objFun_val2)/math.sqrt(sd2)
        
        objFun_val = objFun_val1 + objFun_val2
            
        print(objFun_val)
    
        # Calcualte the root of the integral error
        #objFun_val = math.sqrt(objFun_val/(totalTime-startTime))
        
    except:
        objFun_val = -1
    
    return objFun_val
    

def simulation(s, direc1, direc2):
    
    # Run the simulation with the given initial values and output increment
    dymola.simulateExtendedModel(
    problem=path,
    startTime=0.0,
    stopTime=totalTime,
    outputInterval=1,
    method="Dassl",
    tolerance=0.0001,
    resultFile=direc1,
    initialNames=["heatPump.dataTable.tableQdot_con[2, 2]", "heatPump.dataTable.tableQdot_con[2, 3]",
    "heatPump.dataTable.tableQdot_con[3, 2]", "heatPump.dataTable.tableQdot_con[3, 3]", "sourceTemperature.startTime"],
    initialValues=s
    )
        
    # Get the simulation result
    sim = SimRes(direc2)

    # Extract trajectory 
    values_res = np.column_stack((sim[name_cv1].values(),sim[name_cv2].values()))
    
    return values_res

# Generate reference trajectory
values_ref = simulation([4800, 6300, 4400, 5750, 1200], dir_res_stor_ref, dir_res_open_ref)

# Define the initial conditions for the optimization
s0 = [2000, 2000, 2000, 2000,400]

# Define the boundary conditions for the optimization
#b = ((500,10000),(500,10000),(500,10000),(500,10000))
b = ((500,10000),(500,10000),(500,10000),(500,10000),(0,3600))

# Perform a minimization using the defined obejctive function and frequnecies 
# as decision variables
sol = minimize(objective,s0,method='SLSQP',bounds=b,options={'maxiter':1000, 'eps':200, 'ftol':0.01})
#test = objective(s0)

values_final = simulation([sol.x[0],sol.x[1],sol.x[2],sol.x[3],sol.x[4]], dir_res_stor, dir_res_open)


if dymola is not None:
    dymola.close()
    dymola = None