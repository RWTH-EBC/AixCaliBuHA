import os,sys
sys.path.insert(0, os.path.join('C:\Program Files (x86)\Dymola 2018',
                    'Modelica',
                    'Library',
                    'python_interface',
                    'dymola.egg'))

import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sklearn
from dymola.dymola_interface import DymolaInterface
from dymola.dymola_exception import DymolaException
from modelicares import simres as sr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
from datetime import datetime
import scipy.optimize as opt
import ebcpython.modelica.postprocessing.simres_ebc as sr_ebc


class dymCalibrator():
    def __init__(self, modelName,tunerParams,simSetup,cwdir,packages,qualMeas = "NRMSE"):
        """Calibrates and validates based a given modelName with the tunerParams dictionary and the simulationSetup.
        qualMeas defines the quality measurement."""
        #Initialize the given parameters for the class dymCalibrator
        self.simSetup = simSetup
        self.simSetup["initialNames"] = list(tunerParams) #
        self.modelName = modelName
        self.tunerParams = tunerParams
        self.cwdir = cwdir
        self.packages = packages #Define packages needed for calibration
        #Optional parameters:
        self.qualMeas = qualMeas
        self.printStat = True
        #Define some arrays
        self.counter = 0 #Used to inform about the number of simulations
        self.real_bounds = []
        self.startSet = []
        for key, value in self.tunerParams.items():
            self.real_bounds.append({"uppBou":value["uppBou"],"lowBou":value["lowBou"]})
            self.startSet.append(value["start"]/(value["uppBou"]-value["lowBou"]))
        #Used for to_pandas_ebc. This way the data will get stored in a dataFrame.
        self.aliases = {"heatPump.sigBusHP.T_ret_co": "T_ret_co",
                       "heatPump.sigBusHP.T_ret_ev": "T_ret_ev",
                       "heatPump.sigBusHP.m_flow_co": "m_flow_co",
                       "totalEffPower.y":"Pel_total",
                        "goal_T_ret_co.y":"goal_T_ret_co",
                        "goal_T_ret_ev.y": "goal_T_ret_ev",
                        "goal_Pel_total.y": "goal_Pel_total",
                        "goal_m_flow_co.y":"goal_m_flow_co"}
        #The naming of this list depends on the aliases given.
        self.goals = ["T_ret_co","T_ret_ev","Pel_total"]
        #Setup Dymola and load packages.
        self.setupDym()

###Dymola related funtions
    def setupDym(self):
        self.dymola = DymolaInterface()
        self.dymola.cd(self.cwdir)
        for pack in self.packages:
            print("Loading Model %s" % os.path.dirname(pack).split("\\")[-1])
            res = self.dymola.openModel(pack, changeDirectory=False)
            if not res:
                print(self.dymola.getLastErrorLog())
        print("Loaded modules")

    def simulate(self, initialValues):
        """Simulate given initial values. If simulation terminates without an error, the relevant files are moved to a folder and the data is trimmed."""
        #Just for information sake
        if self.printStat:
            print(initialValues)
        res = self.dymola.simulateExtendedModel(self.modelName,
                                                 startTime=self.simSetup['startTime'],
                                                 stopTime=self.simSetup['stopTime'],
                                                 numberOfIntervals=self.simSetup['numberOfIntervals'],
                                                 outputInterval=self.simSetup['outputInterval'],
                                                 method=self.simSetup['method'],
                                                 tolerance=self.simSetup['tolerance'],
                                                 fixedstepsize=self.simSetup['fixedstepsize'],
                                                 resultFile=self.paramSetString,
                                                 initialNames=self.simSetup['initialNames'],
                                                 initialValues=initialValues)
        if not res[0]:
            print("Failed tuner Params. Trying new ones!")
            print(self.dymola.getLastErrorLog())
            return False
        else:
            new_path = os.path.join(self.cwdir, self.paramSetString)
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            for filename in ["%s.mat"%self.paramSetString, "dslog.txt"]:
                if os.path.isfile(os.path.join(new_path, filename)):
                    os.remove(os.path.join(new_path, filename))
                os.rename(os.path.join(self.cwdir, filename), os.path.join(new_path, filename))
                if filename.endswith(".mat"):
                    self.getGoalData(os.path.join(new_path, filename))
            return True

    def getGoalData(self, filepath):
        """Extracts the relevant data of the simulation result based on the given aliases"""
        if filepath.endswith(".mat"):
            sim = sr.SimRes(filepath)
        else:
            return
        self.curDataFrame = sr_ebc.to_pandas_ebc(sim, names=list(self.aliases),aliases=self.aliases)

###Calibration functions
    def calibrate(self):
        """Using the method "L-BFGS-B", the bounds will always be between 0 and 1, as all parameters are normalized to a value between 0 and 1.
        The kwargs options can be altered to define the accuracy. I used eps=0.01 as a value"""
        options = {"disp":False,
                    "ftol":2.220446049250313e-09,
                    "eps":1e-8
                   }
        res = opt.minimize(self.testNewSet, np.array(self.startSet), method="L-BFGS-B",bounds=opt.Bounds(np.zeros(len(self.startSet)),np.ones(len(self.startSet))), options=options)
        return res #The results can be used to access the minimal values etc.


    def testNewSet(self, newSet):
        initialValues=[] #Convert set to real parameters for modelica.
        for i in range(0,len(newSet)):
            initialValues.append(newSet[i]*(self.real_bounds[i]["uppBou"]-self.real_bounds[i]["lowBou"]))
        self.paramSetString = str(self.counter) #Used to save the data
        if self.printStat:
            print(self.counter)
        res = self.simulate(initialValues) #Simulate the given initial values
        self.counter += 1
        if res:
            self.calcStatValues() #Get the current statistical values for the result
            total = 0
            for goalName, statDic in self.statisticalValues.items():
                # A new definition of the goals is necessary. This will make the weigthing and so on much easier.
                if goalName=="T_ret_ev":
                    total += statDic[self.qualMeas]*0.2
                else:
                    total += statDic[self.qualMeas]*0.4
                #This is also not necessary.
                #print("%s of %s: %s"%(self.qualMeas,goalName,statDic[self.qualMeas]))
            if self.printStat:
                print("Total weigthed %s: %s"%(self.qualMeas,total))
            return total
        else:
            #Punish the failure of the simulation with an extremly high return value
            return 10000000


    def calcStatValues(self):
        """Create a dictionary with every statistical value for every goal-parameter."""
        self.statisticalValues = {}
        for goal in self.goals:
            for colName in list(self.curDataFrame):
                if colName.startswith("goal_%s"%goal):
                    self.exp = self.curDataFrame[colName]
                elif colName.startswith(goal):
                    self.sim = self.curDataFrame[colName]
            tempStatisticalValues = {}
            tempStatisticalValues["MAE"] = mean_absolute_error(self.exp,self.sim)
            tempStatisticalValues["RMSE"] = np.sqrt(mean_squared_error(self.exp,self.sim))
            tempStatisticalValues["R2"] = 1-r2_score(self.exp,self.sim)
            if np.mean(self.exp)!= 0:
                tempStatisticalValues["CVRMSE"] = tempStatisticalValues["RMSE"] / np.mean(self.exp)
            else:
                if self.qualMeas == "CVRMSE":
                    raise ValueError("The experimental gathered data has a mean of zero over the given timeframe.The CVRMSE can not be calculated. Please use the NRMSE")
                else:
                    tempStatisticalValues["CVRMSE"] = 1e10 #Punish the division by zero
            if (np.max(self.exp)-np.min(self.exp)) != 0:
                tempStatisticalValues["NRMSE"] = tempStatisticalValues["RMSE"]/(np.max(self.exp)-np.min(self.exp))
            else:
                if self.qualMeas == "NRMSE":
                    raise ValueError("The experimental gathered data is constant over the given timeframe. The NRMSE can not be calculated. Please use the CVRMSE")
                else:
                    tempStatisticalValues["NRMSE"] = 1e10 #Punish the division by zero
            self.statisticalValues[goal] = tempStatisticalValues

###Validation functions. Just the basics, nothing good yet.
    def validateWithTestParams(self, initialValues):
        """Re-run the testNewSet-Function to validate the given initialValues."""
        x = self.calculateSet(initialValues)
        self.testNewSet(x)

    def calculateSet(self, initialValues):
        """Convert given initial Values to normalized set of parameters for the function testNewSet"""
        x=[]
        for i in range(0,len(initialValues)):
            x.append(initialValues[i]/(self.real_bounds[i]["uppBou"]-self.real_bounds[i]["lowBou"]))
        return np.array(x)

if __name__=="__main__":
    modelName = "HPSystemSimulation.Calibration.HeatPump"
    tunerParams = {"heatPump.GCon":{"start": 10,"uppBou": 50,"lowBou": 0},
                   "heatPump.GEva":{"start": 10,"uppBou": 50,"lowBou": 0},
                   "heatPump.dpCon_nominal":{"start": 20000,"uppBou": 70000,"lowBou": 10000},
                   "heatPump.dpEva_nominal":{"start": 20001,"uppBou": 70000,"lowBou": 10000},
                   "heatPump.CCon":{"start": 100,"uppBou": 100000,"lowBou": 1},
                   "heatPump.CEva":{"start": 100,"uppBou": 100000,"lowBou": 1},
                   "heatPump.GConIns":{"start": 25,"uppBou": 50,"lowBou": 0},
                   "heatPump.GEvaIns":{"start": 25,"uppBou": 50,"lowBou": 0},
                   "heatPump.VCon":{"start": 0.004,"uppBou": 0.01,"lowBou": 0.000001},
                   "heatPump.VEva":{"start": 0.004,"uppBou": 0.01,"lowBou": 0.000001},
                   "heatPump.mFlow_conNominal":{"start": 0.5,"uppBou": 1,"lowBou": 0.1},
                   "heatPump.mFlow_evaNominal": {"start": 0.5, "uppBou": 1, "lowBou": 0.1},
                   "heatPump.refIneFre_constant":{"start": 0.01,"uppBou": 0.5,"lowBou": 0.0001},
                   "heatPump.tauHeaTra":{"start": 3000,"uppBou": 5000,"lowBou": 1200},
                   "heatPump.TConStart":{"start": 292.15,"uppBou": 340,"lowBou": 283.15},
                   "heatPump.TEva_start": {"start": 278.15, "uppBou": 300, "lowBou": 273.15}}
    simSetup = {'startTime': 0.0,
                'stopTime':5400,
               'numberOfIntervals': 0,
                'outputInterval':1,
               'method': 'Dassl',
               'tolerance': 0.0001,
               'fixedstepsize': 0.0,
               'resultFile': 'resultFile',
               'autoLoad': None,
               'initialNames':list(tunerParams)}
    cwdir = r"D:\04_pyGit\Bachelorarbeit\07_Kalibrierung\02_Rohdaten"
    packages = [r"D:\02_git\AixLib_cal\AixLib\package.mo",
                r"D:\02_git\HPSystemSimulation\HPSystemSimulation\package.mo"]
    dymCal = dymCalibrator(modelName, tunerParams,simSetup,cwdir,packages)
    dymCal.calibrate()
    dymCal.dymola.close()
    dymCal.genResults()