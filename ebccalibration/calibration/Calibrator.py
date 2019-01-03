"""
Script for setting up the calibrator-class. This class contains the goal-function, the
calculation of the statistical values and minimizer
"""
import scipy.optimize as opt
import sklearn.metrics as skmetrics
import numpy as np
import ebcpython.modelica.postprocessing.simres_ebc as sr_ebc
from modelicares import simres as sr
from datetime import datetime #Used for saving of relevant files
import xml.etree.ElementTree as ET
from xml.dom import minidom
import matplotlib.pyplot as plt
import os, dicttoxml, re

class calibrator():
    def __init__(self, goals, tunerPara, qualMeas, method, dymAPI, bounds = None, timeInfoTupel = None, **kwargs):
        """
        Class for a calibrator.
        :param goals: list
        Dictionary with information about the goal-variables. Each goal has a sub-dictionary.
            meas: str
            Name of measured data in dataframe
            sim: str
            Name of simulated data in dataframe
            meas_full_modelica_name:
            Name of the measured data in modelica
            sim_full_modelica_name:
            Name of the simulated data in modelica
            weighting: float
            Weighting for the resulting objective function
        :param tunerPara: dict
        Dictionary with information about tuner parameters.
            key: str
            initial name of tuner parameter
            value: dict
                start: float, int
                start-value
                uppBou: float, int
                upper boundary for calibration
                lowBou: float, int
                lower boundary for calibration
        :param qualMeas: str
        Statistical value for the objective function, e.g. RMSE, MAE etc.
        :param method: str
        Used method for minimizer. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html for more info
        :param dymAPI: dymolaInterface
        Class for executing a simulation in dymola
        :param bounds: optimize.Bounds object, Default: None
        Used if the boundaries differ from 0 and 1 (already scaled / normalized)
        :param timeInfoTupel: tupel
        Tupel with startTime and entTime info special sections of the resulting array are of interest for the objective function.
        :param kwargs: dict
            'method_options': dict
            Optional class parameters
                key: method_options: str
                value: dict
                Dictionary containing optional parameters for the minimizer. See scipy.optimize.minimize documentation for detailed info.
            'tol': float or None
            if objective <= tol, the calibration stops
            'plotCallback': bool
            Whether to plot the current status or not

            See list 'booleankwargs' for all kwargs of type boolean.
        """
        if self._checkGoals(goals):
            self.goals = goals
        if self._checkTunerParas(tunerPara):
            self.tunerPara = tunerPara
        self.initalSet = [] #Create inital set for first iteration
        self.bounds_scaled = [] #Create a list to denormalize the sets for the simulation
        for key, value in tunerPara.items():
            self.bounds_scaled.append({"uppBou": value["uppBou"],
                                       "lowBou": value["lowBou"]})
            self.initalSet.append((value["start"]-value["lowBou"])/(value["uppBou"]-value["lowBou"]))
        if not bounds:
            self.bounds = opt.Bounds(np.zeros(len(self.initalSet)), np.ones(len(self.initalSet))) #Define boundaries for normalized values, e.g. [0,1]
        #Set other values
        self.method = method
        self.qualMeas = qualMeas
        #Create bounds based on the dictionary.
        self.dymAPI = dymAPI
        #Create aliases dict out of given goals-dict.
        self.aliases = {}
        for goal in goals:
            self.aliases[goal["sim_full_modelica_name"]] = goal["sim"]
            self.aliases[goal["meas_full_modelica_name"]] = goal["meas"]
        self.dymAPI.simSetup["initialNames"] = list(self.tunerPara)
        #Starttime and endtime
        assert len(timeInfoTupel) == 2, "The lenght of the timeInfoTupel has to be equal to two. First entry the starttime, second entry the endtime"
        self.timeInfoTupel = timeInfoTupel
        if self.timeInfoTupel:
            self.startTime = self.timeInfoTupel[0]
            self.endTime = self.timeInfoTupel[1]

        #kwargs
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)
        if not hasattr(self, "method_options"):
            self.method_options = {}
        if not hasattr(self, "tol"):
            self.tol = None
        booleankwargs = ["plotCallback", "saveFiles", "continouusCalibration"]
        for bool in booleankwargs:
            if not hasattr(self, bool):
                setattr(self, bool, False)
        #Set counter for number of iterations of the objective function
        self.counter = 0
        self.objHis = []
        self.counterHis = []
        self.last_set = np.array([]) # Used if simulation failes
        self.log = ""
        if self.continouusCalibration:
            self.totalMin = {"obj": 1e10,
                             "dsfinal":"",
                             "dsres":""}
            self.savepathMinResult = os.path.join(self.dymAPI.cwdir, "totalMin")
            if not os.path.isdir(self.dymAPI.cwdir):
                print("Creating working directory {}".format(self.dymAPI.cwdir))
                os.mkdir(self.dymAPI.cwdir)
            if not os.path.isdir(self.savepathMinResult):
                os.mkdir(self.savepathMinResult)

    def calibrate(self, obj):
        """
        Optimizes the given inital set of parameters with the objective function.
        :param obj: callable
        Objective function for the minimize function
        :return: OptimizeResult
        Result of the optimization. Most important: x and fun
        """
        infoString = self._getNameInfoString()
        print(infoString)
        self.log += "\n" + infoString
        try:
            res = opt.minimize(fun=obj, x0=np.array(self.initalSet), method=self.method, bounds=self.bounds, tol=self.tol, options=self.method_options)
            if not self.continouusCalibration:
                self.dymAPI.dymola.close() # Close dymola
            self.save_log_as_csv()
            return res
        except Exception as e:
            print("Parameter set which caused failed simulation:")
            print(self._getValInfoString(self.last_set, forLog = True))
            self.dymAPI.dymola.close()
            raise Exception(e) # Actually raise the error so the script stops.

    def objective(self, set):
        """
        Default objective function.
        The usual function will be implemented here:
        1. Convert the set to modelica-units
        2. Simulate the converted-set
        3. Get data as a dataFrame
        4. Calculate the objective based on statistical values
        :param set: np.array
        Array with normalized values for the minimizer
        :return:
        Objective value based on the used quality measurement
        """
        self.last_set = set
        self.counter += 1 #Increase counter
        conv_set = self._convSet(set) #Convert set if multiple goals of different scales are used
        self.dymAPI.set_initialValues(conv_set) #Set initial values
        saveName = "%s"%str(self.counter) #Generate the folder name for the calibration
        success, filepath, strucParams = self.dymAPI.simulate(saveFiles=self.saveFiles, saveName=saveName, getStructurals=True) #Simulate
        if success:
            self.dymAPI.strucParams = strucParams
            df = self.get_trimmed_df(filepath, self.aliases, getTrajectorieNames = self.continouusCalibration) #Get results
            if self.timeInfoTupel:
                # Trim results based on start and endtime
                df = df[self.startTime:self.endTime]
        total_res = 0
        for goal in self.goals:
            goal_res = self.calc_statValues(df[goal["meas"]],df[goal["sim"]])
            total_res += goal["weighting"] * goal_res[self.qualMeas]
        self.objHis.append(total_res)
        self.counterHis.append(self.counter)
        self.callbackF(set)
        if self.continouusCalibration:
            if self.totalMin["obj"] > total_res:
                self.totalMin = {"obj":total_res,
                                 "dsres": filepath,
                                 "dsfinal":os.path.join(os.path.dirname(filepath), "dsfinal.txt")}
                #Overwrite old results:
                if os.path.isfile(os.path.join(self.savepathMinResult, "dsres.mat")):
                    os.remove(os.path.join(self.savepathMinResult, "dsres.mat"))
                if os.path.isfile(os.path.join(self.savepathMinResult, "dsfinal.txt")):
                    os.remove(os.path.join(self.savepathMinResult, "dsfinal.txt"))
                os.rename(filepath, os.path.join(self.savepathMinResult, "dsres.mat"))
                os.rename(os.path.join(os.path.dirname(filepath), "dsfinal.txt"), os.path.join(self.savepathMinResult, "dsfinal.txt"))
        return total_res

    def callbackF(self, xk):
        """
        Default callback function for when the objective function of this class is used.
        Either plots the current status or prints it to the
        :param set:
        Array with normalized values for the minimizer
        :return:
        None
        """
        infoString = self._getValInfoString(xk)
        print(infoString)
        self.log += "\n" + infoString
        if self.plotCallback:
            if not hasattr(self, "fig"): # Instanciate the figure and ax
                self.fig, self.ax = plt.subplots(1, 1)
            self.ax.plot(self.counterHis[-1], self.objHis[-1], "ro")
            self.ax.set_ylabel(self.qualMeas)
            self.ax.set_xlabel("Number iterations")
            self.ax.set_title(self.dymAPI.modelName)
            #If the changes are small, it seems like the plot does not fit the printed values. This boolean assures that no offset is used.
            self.ax.ticklabel_format(useOffset=False)
            plt.draw()
            plt.pause(1e-5)

    def _convSet(self, set):
        """
        Convert given set to initial values in modelica according to function:
        iniVal_i = set_i*(max(iniVal_i)-min(iniVal_i)) + min(iniVal_i)
        :param set:
        Array with normalized values for the minimizer
        :return:
        List of inital values for dymola
        """
        initialValues = []
        for i in range(0, len(set)):
            initialValues.append(set[i]*(self.bounds_scaled[i]["uppBou"] - self.bounds_scaled[i]["lowBou"]) + self.bounds_scaled[i]["lowBou"])
        return initialValues


    def get_trimmed_df(self, filepath, aliases, getTrajectorieNames = False):
        """
        Create a dataFrame based on the given result-file and the aliases with the to_pandas_ebc function.
        :param filepath: str, os.path.normpath
        Path to the matfile
        :param aliases: dict
        Dictionary for selecting and naming the dataframe columns
        :return: df: pandas.DataFrame
        Data Frame object with relevant data
        """
        if not filepath.endswith(".mat"):
            raise TypeError("Given filename is not of type *.mat")

        sim = sr.SimRes(filepath)
        df = sr_ebc.to_pandas_ebc(sim, names=list(aliases), aliases=aliases, useUnit = False)
        if getTrajectorieNames:
            self.trajNames = sr_ebc.get_trajectories(sim)
        return df


    def calc_statValues(self, meas, sim):
        """
        Calculates multiple statistical values for the given numpy array of measured and simulated data.
        Calculates:
        MAE(Mean absolute error), RMSE(root mean square error), R2(coefficient of determination), CVRMSE(variance of RMSE), NRMSE(Normalized RMSE)
        :param meas:
        Array with measurement data
        :param sim:
        Array with simulation data
        :return: statValues: dict
        Containing all calculated statistical values
        """
        statValues = {}
        statValues["MAE"] = skmetrics.mean_absolute_error(meas, sim)
        statValues["RMSE"] = np.sqrt(skmetrics.mean_squared_error(meas, sim))
        statValues["R2"] = 1 - skmetrics.r2_score(meas, sim)
        if np.mean(meas) != 0:
            statValues["CVRMSE"] = statValues["RMSE"] / np.mean(meas)
        else:
            if self.qualMeas == "CVRMSE":
                raise ValueError(
                    "The experimental gathered data has a mean of zero over the given timeframe.The CVRMSE can not be calculated. Please use the NRMSE")
            else:
                statValues["CVRMSE"] = 1e10  # Punish the division by zero
        if (np.max(meas) - np.min(meas)) != 0:
            statValues["NRMSE"] = statValues["RMSE"] / (np.max(meas) - np.min(meas))
        else:
            if self.qualMeas == "NRMSE":
                raise ValueError(
                    "The experimental gathered data is constant over the given timeframe. The NRMSE can not be calculated. Please use the CVRMSE")
            else:
                statValues["NRMSE"] = 1e10  # Punish the division by zero
        return statValues

    def save_result(self, res, savepath, ftype = "svg"):
        """
        Process the result, re-run the simualation and generate a logFile for the minimal quality measurement
        :param res: minimize.result
        Result object of the minimization
        :param savepath: str, os.path.normpath
        Directory where to store the results
        :param ftype: str
        svg, pdf or png
        """
        result_log = "Results for calibration of model: %s\n"%self.dymAPI.modelName
        result_log += "Minimal %s: %s\n"%(self.qualMeas,res.fun)
        result_log += "Final parameter values:\n"
        result_log += "%s\n"%self._getNameInfoString(forLog=True)
        result_log += "%s\n"%self._getValInfoString(res.x, forLog=True)
        result_log += "Number of iterations: %s\n"%self.counter
        result_log += "\nIteration log:\n" + self.log
        datestring = datetime.now().strftime("%Y%m%d_%H%M%S")
        f = open(os.path.join(savepath, "CalibrationLog_%s.txt"%datestring), "a+")
        f.write(result_log)
        f.close()
        if self.plotCallback:
            plt.savefig(os.path.join(savepath, "iterationPlot.%s"%ftype))

    def save_log_as_csv(self, sep = ";"):
        """
        Saves the log-string as a csv-file
        :param sep: str
        Seperator used in csv
        """
        savepath = os.path.join(self.dymAPI.cwdir, "log.csv")
        lines = self.log.split("\n")
        lines_csv = [re.sub("\s+", sep, line.strip()) for line in lines]
        f = open(savepath, "a+")
        f.seek(0)
        f.truncate()
        f.write("\n".join(lines_csv[1:]))
        f.close()


    def _checkGoals(self, goals):
        """
        Checks the given goals-list for correct formatting
        :param goals: list
        List containing dictionary with goals
        :return: bool: True if success
        """
        if type(goals) != type([]) or len(goals) == 0: #Has to be type list and contain at least one entry
            raise TypeError("Given goal list is not of type list or is empty")
        total_weighting = 0
        for goal in goals:
            if not ("meas" in goal and "sim" in goal and "weighting" in goal and "meas_full_modelica_name" in goal and "sim_full_modelica_name" in goal and len(goal.keys())==5):
                raise Exception("Given goal dict is no well formatted.")
            else:
                total_weighting += goal["weighting"]
        if total_weighting != 1:
            raise Exception("Given combiation of weightings does not euqal 1")
        return True #If no error has occured, this check passes
    def _checkTunerParas(self, tunerPara):
        """
        Checks given tuner-para dictionary on correct formatting
        :param tunerPara:
        Dictionary with information about tuner parameters. name, start-value, upper and lower-bound
        :return: bool: True on success
        """
        if type(tunerPara) != type({}) or len(tunerPara) == 0: #Has to be type list and contain at least one entry
            raise TypeError("Given tuner parameters dictionary is not of type dict or is empty")
        for key, para_dict in tunerPara.items():
            if type(key) != type(""):
                raise TypeError("Parameter name %s is not of type string"%str(key))
            if not ("start" in para_dict and "uppBou" in para_dict and "lowBou" in para_dict and len(para_dict.keys())==3):
                raise Exception("Given tunerPara dict %s is no well formatted."%para_dict)
            if para_dict["uppBou"] - para_dict["lowBou"] <= 0:
                raise ValueError("The given upper boundary is less or equal to the lower boundary.")
        return True #If no error has occured, this check passes

    def _getNameInfoString(self, forLog = False):
        """
        Returns a string with the names of current tunerParameters
        :param forLog: Boolean
        If the string is created for the final log file, the best obj is not of interest
        :return: str
        The desired string
        """
        initialNames = list(self.tunerPara.keys())
        if not forLog:
            infoString = "{0:4s}".format("Iter")
        else:
            infoString = ""
        for i in range(0, len(initialNames)):
            infoString += "   {0:9s}".format(initialNames[i])

        if not forLog:
            infoString += "   {0:9s}".format(self.qualMeas)
        else:
            infoString = infoString[3:]
        return infoString

    def _getValInfoString(self, set, forLog = False):
        """
        Returns a string with the values of current tunerParameters
        :param set: np.array
        Array with the current values of the calibration
        :param forLog: Boolean
        If the string is created for the final log file, the best obj is not of interest
        :return: str
        The desired string
        """
        iniVals = self._convSet(set)
        if not forLog:
            infoString = '{0:4d}'.format(self.counter)
        else:
            infoString = ""
        for i in range(0, len(iniVals)):
            infoString += "   {0:3.6f}".format(iniVals[i])
        # Add the last return value of the objective function.
        if not forLog:
            infoString += "   {0:3.6f}".format(self.objHis[-1])
        else:
            infoString = infoString[3:]
        return infoString

###General functions:
def load_tuner_xml(filepath):
    """
    Load the tuner parameters dictionary from the given xml-file
    :param filepath: str, os.path.normpath
    :return: tunerPara: dict
    Dictionary with information about tuner parameters. name, start-value, upper and lower-bound
    """
    root = _loadXML(filepath)
    tunerPara = {}
    for initalName in root:
        tempTunerPara = {}
        for elem in initalName:
            tempTunerPara[elem.tag] = float(elem.text) #All values inside the dict are floats
        tunerPara[initalName.tag] = tempTunerPara
    return tunerPara

def load_goals_xml(filepath):
    """
    Load the goals from the given xml-file
    :param filepath: str, os.path.normpath
    :return: goals: list
    List of dictionaries containing the goals
    """
    root = _loadXML(filepath)
    goals = []
    for goalElem in root:
        goal = {}
        for elem in goalElem:
            if elem.tag == "weighting":
                goal[elem.tag] = float(elem.text)
            else:
                goal[elem.tag] = elem.text
        goals.append(goal)
    return goals

def saveXML(filepath, object):
    """
    Saves a root in a readable str-xml file
    :param filepath: str, os.path.normpath
    :param object: list, dict
    Either list of goals or dict of tuner parameters
    :return: None
    """
    root = dicttoxml.dicttoxml(object)
    xmlstr = minidom.parseString(root).toprettyxml(indent="   ")
    f = open(filepath, "w")
    f.write(xmlstr)
    f.close()

def _loadXML(filepath):
    """
    Loads any xml file
    :param filepath: str, os.path.normpath
    :return:
    root of the xml
    """
    tree = ET.parse(filepath)  # Get tree of XML-file
    return tree.getroot()  # The root holds all plot-elements

def join_tunerParas(continouusData):
    """
    Join all initialNames used for calibration in the given dataset.
    :param continouusData: list
    Contains dictionaries with goals, tunerParas etc.
    :return: list
    Joined list
    """
    joinedList = []
    for c in continouusData:
        for name in c["tunerPara"].keys():
            if name not in joinedList:
                joinedList.append(name)
    return joinedList

def alterTunerParas(newTuner, calHistory):
    """
    Based on old calibration results, this function alters the start-values for the new tunerPara-Set
    :param newTuner: dict
    Tuner Parameter dict for the next time-interval
    :param calHistory: list
    List with all results and data from previous calibrations
    :return: tunerParaDict: dict
    Dictionary with the altered tunerParas
    """
    newStartDict = _get_continouusAverages(calHistory)
    for key, value in newTuner.items():
        if key in newStartDict: #Check if the parameter will be used or not.
            newTuner[key]["start"] = newStartDict[key]
    return newTuner

def _get_continouusAverages(calHistory):
    """
    Function to get the average value of a tuner parameter based on time- average
    :param calHistory: list
    List with all results and data from previous calibrations
    :return: newStartDict: dict
    Dictionary with the average start parameters
    """
    #Iterate over calibration historie and create a new dictionary with the start-values for a given initialName#
    newStartDict = {}
    totalTime = 0
    for calHis in calHistory:
        resIniVals = calHis["cal"]._convSet(calHis["res"].x)
        timedelta = calHis["continouusData"]["stopTime"] - calHis["continouusData"]["startTime"]
        totalTime += timedelta
        tunerPara = calHis["continouusData"]["tunerPara"]
        for i in range(0,len(tunerPara.keys())):
            iniName = list(tunerPara.keys())[i]
            if iniName in newStartDict:
                newStartDict[iniName] += resIniVals[i] * timedelta
            else:
                newStartDict[iniName] = resIniVals[i] * timedelta
    for key, value in newStartDict.items():
        newStartDict[key] = value/totalTime #Build average again
    return newStartDict