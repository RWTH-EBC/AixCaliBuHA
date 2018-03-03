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

import BAC0
import time 

import matplotlib.pyplot as plt
import pickle


base_path = r'D:\Testing\PIDController'

# AixLib directory
dir_Aix = os.path.join(r'D:\Git\AixLib\AixLib\package.mo')

# Use this string to construct the directory where to store the simulation result
dir_res_stor = base_path + r'\dsres'

# Use this path to open the simulation results
dir_res_open = base_path + r'\dsres.mat'

# Name of the first controller output
name_c1 = 'pIDControllerJCIInterval.Out'

# Name of the second controller output
name_c2 = 'pIDControllerJCI.Out'

# Name of the second controller output
name_c3 = 'conPID.y'

os.chdir(base_path)

def start_bacnet():
    bacnet = BAC0.connect('137.226.249.10')
    # or specify the IP you want to use / bacnet = BAC0.connect(ip='192.168.1.10')
    # by default, it will attempt an internet connection and use the network adapter
    # connected to the internet.
    
    # Define a controller (this one is on MSTP #3, MAC addr 4, device ID 5504)
    mycontroller = BAC0.device('137.226.249.116', 74, bacnet)
    
    return mycontroller

# Create the objective function that will simulate the model with the respective
# inputs and return the obejctive function value calculated by the model. 
# The tuple s contains the current values of the decision variables
def signals(s):  
    
    # Use this vector to create the first input signal. The chirp signal 
    # becomes a sine wave, whereas the frequnecy is a decision variable
    vect_1 = (20+4*(signal.chirp(t,s[0], 1, s[0]).reshape((1,totalTime))))
    arr_1 = np.append(t, vect_1, axis=0)
    arr_trans = np.transpose(arr_1)
    sio.savemat('CV.mat', {'CV':arr_trans})
    
    # Use this vector to create the second input signal. The chirp signal 
    # becomes a sine wave, whereas the frequnecy is a decision variable
    vect_2 = (23+5*(signal.chirp(t,s[1], 1, s[1]).reshape((1,totalTime))))
    arr_2 = np.append(t, vect_2, axis=0)
    arr_trans_2 = np.transpose(arr_2)
    sio.savemat('set.mat', {'set':arr_trans_2})

    # Use this array to store a .mat file that can be read by Dymola. This 
    # file inputs the trajectory of the flow temperature
    arr = np.append(vect_1, vect_2, axis=0)
    
    #return the negative value for maximization of the error (required for
    # minimization)
    return arr

incr = 10
# Simulation time
totalTime=incr*180

# Create an array containing the time stamp
t = np.linspace(0, totalTime, totalTime)
t = t.reshape(1,totalTime)

#Prepare the communication with the real controller
s = [1/600, 1/1000]
arr = signals(s)
#pos = np.empty([int(totalTime/incr)])


#mycontroller = start_bacnet()

'''
for t in range(0,totalTime,incr):
    str_list = str(mycontroller['4120.BOI.VAL.open'])
    num_str = str_list.split(' ')
    pos[int(t/incr)] = float(num_str[2])
    mycontroller['4120.BOI.T.FLOW'] = arr[0][t]
    mycontroller['4120.BOI.T.FLOW.SET'] = arr[1][t]
    time.sleep(incr)
'''    

#Prepare the simulation of the virtual controllers
cmd = base_path + r'\dymosim.exe'
subprocess.run(cmd, stdout=subprocess.PIPE)  

# Get the simulation result
sim = SimRes(dir_res_open)

# Extract the trajectory of the controlled variable
values_c1 = sim[name_c1].values() 

# Extract the trajectory of the set point variable
values_c2 = sim[name_c2].values() 

# Extract the trajectory of the set point variable
values_c3 = sim[name_c3].values() 

t2 = np.linspace(0, 1800, 180)

fig = plt.figure()
fig.patch.set_alpha(1)
ax = fig.add_subplot(111)
ax.patch.set_alpha(1)

line1 = ax.plot(t2,values_c1, '-.', label='JCI Intervall 15 s simuliert')
line2 = ax.plot(t2,values_c3, label='JCI Intervall 1 s simuliert')
line3 = ax.plot(t2,values_c3, '--', label='AixLib kontinuierlich simuliert')
line4 = ax.plot(t2,pos, '--', label='Belimo emuliert')
ax.set_xlabel('Zeit in s')
ax.set_ylabel('Ventilöffnung in %')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol = 2)
plt.savefig('ValveOpening.svg')
