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
    def __init__(self, modelName,tunerParams):
        self.simSetup = {'startTime': 0.0,
                        'stopTime':5400,
                       'numberOfIntervals': 0,
                        'outputInterval':1,
                       'method': 'Dassl',
                       'tolerance': 0.0001,
                       'fixedstepsize': 0.0,
                       'resultFile': 'resultFile',
                       'autoLoad': None,
                       'initialNames':list(tunerParams)}
        self.modelName = modelName
        self.tunerParams = tunerParams
        self.bounds,self.real_bounds = [],[]
        self.history = []
        self.startSet = []
        for key, value in self.tunerParams.items():
            self.bounds.append([0,1])
            self.real_bounds.append({"uppBou":value["uppBou"],"lowBou":value["lowBou"]})
            self.startSet.append(value["start"]/(value["uppBou"]-value["lowBou"]))
        self.aliases = {"heatPump.sigBusHP.T_ret_co": "T_ret_co",
                       "heatPump.sigBusHP.T_ret_ev": "T_ret_ev",
                       "heatPump.sigBusHP.m_flow_co": "m_flow_co",
                       "totalEffPower.y":"Pel_total",
                        "goal_T_ret_co.y":"goal_T_ret_co",
                        "goal_T_ret_ev.y": "goal_T_ret_ev",
                        "goal_Pel_total.y": "goal_Pel_total",
                        "goal_m_flow_co.y":"goal_m_flow_co"}
        self.goals = ["T_ret_co","T_ret_ev","Pel_total"]
        self.plotLoc = {"Pel_total":[0,0],
                    "T_ret_co":[1,0],
                    "T_ret_ev":[0,1],
                    "m_flow_co":[1,1]}
        self.packages = [r"D:\02_git\AixLib_cal\AixLib\package.mo",
                         r"D:\02_git\HPSystemSimulation\HPSystemSimulation\package.mo"]
        self.cwdir = r"D:\04_pyGit\Bachelorarbeit\07_Kalibrierung\02_Rohdaten"
        self.setupDym()
        self.failedCounter = 0
        self.counter = 0
        self.min = {"minStat":1000000,
                    "initialValues": [],
                    "df" : 0,
                    "statValues":{}}

    def calibrate(self):
        options = {"disp":True,
                    "ftol":2.220446049250313e-09,
                    "eps":0.01
                   }
        res = opt.minimize(self.testNewSet, np.array(self.startSet), method="L-BFGS-B",bounds=opt.Bounds(np.zeros(len(self.startSet)),np.ones(len(self.startSet))), options=options)
        print(res)
    def validateWithTestParams(self, initialValues):
        x = self.calculateSet(initialValues)
        self.testNewSet(x)
    def calculateSet(self, initialValues):
        x=[]
        for i in range(0,len(initialValues)):
            x.append(initialValues[i]/(self.real_bounds[i]["uppBou"]-self.real_bounds[i]["lowBou"]))
        return np.array(x)
    def testNewSet(self, newSet):
        self.curTunerSet = newSet
        self.paramSetString = str(self.counter)
        print(self.counter)
        res = self.simulate()
        self.counter += 1
        if res:
            self.calcStatValues()
            total = 0
            for goalName, statDic in self.statisticalValues.items():
                if goalName=="T_ret_ev":
                    total += statDic["NRMSE"]*0.2
                else:
                    total += statDic["NRMSE"]*0.4
                print("NRMSE of %s: %s"%(goalName,statDic["NRMSE"]))
            print("Total weigthed NRMSE: %s"%total)
            if total < self.min["minStat"]:
                self.min["minStat"] = total
                self.min["df"] = self.curDataFrame
                self.min["initialValues"] =  self.initialValues
                self.min["statValues"] = self.statisticalValues
            return total
        else:
            self.failedCounter +=1
            return 10000000

    def setupDym(self):
        self.dymola = DymolaInterface()
        self.dymola.cd(self.cwdir)
        for pack in self.packages:
            print("Loading Model %s" % os.path.dirname(pack).split("\\")[-1])
            res = self.dymola.openModel(pack, changeDirectory=False)
            if not res:
                print(self.dymola.getLastErrorLog())
        print("Loaded modules")

    def calcStatValues(self):
        self.statisticalValues = {}
        maxCritDiff = 0
        for goal in self.goals:
            for colName in list(self.curDataFrame):
                if colName.startswith("goal_%s"%goal):
                    self.x = self.curDataFrame[colName]
                elif colName.startswith(goal):
                    self.y = self.curDataFrame[colName]
            e = self.x-self.y
            tempStatisticalValues = {}
            tempStatisticalValues["MAE"] = mean_absolute_error(self.x,self.y)
            tempStatisticalValues["RMSE"] = np.sqrt(mean_squared_error(self.x,self.y))
            tempStatisticalValues["R2"] = 1-r2_score(self.x,self.y)
            if (np.max(self.x)-np.min(self.x)) != 0:
                tempStatisticalValues["NRMSE"] = tempStatisticalValues["RMSE"]/(np.max(self.x)-np.min(self.x))
            else:
                tempStatisticalValues["NRMSE"] = 1000000000
            self.statisticalValues[goal] = tempStatisticalValues

    def simulate(self):
        self.initialValues=[]
        for i in range(0,len(self.curTunerSet)):
            self.initialValues.append(self.curTunerSet[i]*(self.real_bounds[i]["uppBou"]-self.real_bounds[i]["lowBou"]))
        print(self.initialValues)
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
                                                 initialValues=self.initialValues)
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
        if filepath.endswith(".mat"):
            sim = sr.SimRes(filepath)
        else:
            return
        self.curDataFrame = sr_ebc.to_pandas_ebc(sim, names=list(self.aliases),aliases=self.aliases)
        #print("Saved relevant data in dataFrame")

    def genResults(self):
        path = r"D:\04_pyGit\Bachelorarbeit\07_Kalibrierung\03_Ergebnisse"
        ending =  datetime.now().strftime("%Y%m%d_%H%M%S")
        res = ""
        res += "Min NRSME: %s" % self.min["minStat"]
        for val in self.min["initialValues"]:
            res += "\n" + str(val).replace(".", ",")
        for goal, stats in self.min["statValues"].items():
            res += "\n" + goal
            res += "\n" + "MAE;RMSE;R2;NRMSE"
            res += "\n" + ";".join([str(val) for val in list(stats.values())]).replace(".", ",")
        f = open(os.path.join(path, "result_%s.txt"%ending), "a+")
        f.write(res)
        f.close()
        df = self.min["df"]
        fig, axes = plt.subplots(3, 1, sharex=True)
        for key in df.keys():
            if "Pel" in key:
                ax = axes[0]
                ax.set_ylabel("Pel")
                ax.plot(df[key])
            elif "T_ret_co" in key:
                ax = axes[1]
                ax.set_ylabel("T_ret_co")
                ax.plot(df[key])
            elif "T_ret_ev" in key:
                ax = axes[2]
                ax.set_ylabel("T_ret_ev")
                ax.plot(df[key])
        fig.savefig(os.path.join(path, "result_%s.svg"%ending))
if __name__=="__main__":
    modelName = "HPSystemSimulation.Calibration.HeatPump"
    tunerParams = {"heatPump.GCon":{"start": 10,"uppBou": 50,"lowBou": 0},
                   "heatPump.GEva":{"start": 10,"uppBou": 50,"lowBou": 0},
                   "heatPump.dpCon_nominal":{"start": 20000,"uppBou": 70000,"lowBou": 10000},
                   "heatPump.dpEva_nominal":{"start": 20001,"uppBou": 70000,"lowBou": 10000},
                   #"heatPump.CCon":{"start": 100,"uppBou": 100000,"lowBou": 1},
                   #"heatPump.CEva":{"start": 100,"uppBou": 100000,"lowBou": 1},
                   "heatPump.GConIns":{"start": 25,"uppBou": 50,"lowBou": 0},
                   "heatPump.GEvaIns":{"start": 25,"uppBou": 50,"lowBou": 0},
                   "heatPump.VCon":{"start": 0.004,"uppBou": 0.01,"lowBou": 0.000001},
                   "heatPump.VEva":{"start": 0.004,"uppBou": 0.01,"lowBou": 0.000001},
                   #"heatPump.mFlow_conNominal":{"start": 0.5,"uppBou": 1,"lowBou": 0.1},
                   #"heatPump.mFlow_evaNominal": {"start": 0.5, "uppBou": 1, "lowBou": 0.1},
                   "heatPump.refIneFre_constant":{"start": 0.01,"uppBou": 0.5,"lowBou": 0.0001},
                   "heatPump.tauHeaTra":{"start": 3000,"uppBou": 5000,"lowBou": 1200}}
                   #"heatPump.TConStart":{"start": 292.15,"uppBou": 340,"lowBou": 283.15},
                   #"heatPump.TEva_start": {"start": 278.15, "uppBou": 300, "lowBou": 273.15}}
    structuralParams = {"heatPump.nthOrder":{"start": 2,"uppBou": 5,"lowBou": 1, "stepsize":1}}
    dymCal = dymCalibrator(modelName, tunerParams)
    dymCal.calibrate()
    #iniValues = [11.994231781228223, 10.631564896045438, 35879.70138771951, 42094.67744581331, 24.958522274588702, 25.00458049725988, 0.003809317880356165, 0.004031841014591179, 0.0023891887211390443, 2995.3248868969295]
    iniValues = [1.211875502439817, 1.556130302740115, 60000.0, 66000.0, 22.24794652655077, 24.86774236604858, 0.0022409506360725456, 0.004024581855917184, 0.004365884623789393, 2936.88526760581]
    #dymCal.validateWithTestParams(iniValues)
    dymCal.dymola.close()
    dymCal.genResults()