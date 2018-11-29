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

class calibrator():
    def __init__(self, goals, tunerPara, qualMeas, method, dymAPI,aliases, bounds = None):
        """
        Class for a calibrator.
        :param goals: list
        Dictionary with information about the goal-variables. Each goal has a sub-dictionary.
            meas: str
            Name of measured data in dataframe
            sim: str
            Name of simulated data in dataframe
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
        :param aliases:
        :param bounds: optimize.Bounds object, Default: None
        Used if the boundaries differ from 0 and 1
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
            self.initalSet.append(value["start"]/(value["uppBou"]-value["lowBou"]))
        if not bounds:
            self.bounds = opt.Bounds(np.zeros(len(self.initalSet)),np.ones(len(self.initalSet))) #Define boundaries for normalized values, e.g. [0,1]
        self.methodOptions = {"disp":False,
                    "ftol":2.220446049250313e-09,
                    "eps":0.1
                   }
        #Set other values
        self.method = method
        self.qualMeas = qualMeas
        #Create bounds based on the dictionary.
        self.dymAPI = dymAPI
        self.aliases = aliases
        self.dymAPI.set_initialNames(list(self.tunerPara))
        #Set counter for number of iterations of the objective function
        self.counter = 0
        self.startDateTime = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.objHis = []
        self.counterHis = []

    def calibrate(self, obj):
        """
        Optimizes the given inital set of parameters with the objective function.
        :param obj: callable
        Objective function for the minimize function
        :return: OptimizeResult
        Result of the optimization. Most important: x and fun
        """
        res = opt.minimize(obj, np.array(self.initalSet), method=self.method, bounds= self.bounds, options=self.methodOptions, callback=self.callbackF)
        return res

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
        self.counter += 1 #Increase counter
        conv_set = self._convSet(set) #Convert set if multiple goals of different scales are used
        self.dymAPI.set_initialValues(conv_set) #Set initial values
        saveName = "%s_%s"%(self.startDateTime,str(self.counter)) #Generate the folder name for the calibration
        success, filepath = self.dymAPI.simulate(saveFiles=False, saveName=saveName, getStructurals=True) #Simulate
        if success:
            df = self.get_trimmed_df(filepath, self.aliases) #Get results
        else:
            raise Exception("The given bounds or parameters resulted in a failure of the simulation!")
        total_res = 0
        for goal in self.goals:
            goal_res = self.calc_statValues(df[goal["meas"]],df[goal["sim"]])
            total_res += goal["weighting"] * goal_res[self.qualMeas]
        self.objHis.append(total_res)
        self.counterHis.append(self.counter)
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
        plt.plot(self.counterHis, self.objHis)
        plt.draw()
        plt.pause(1e-5)

    def _convSet(self, set):
        """
        Convert given set to initial values in modelica
        :param set:
        Array with normalized values for the minimizer
        :return:
        List of inital values for dymola
        """
        initialValues = []
        for i in range(0, len(set)):
            initialValues.append(set[i]*(self.bounds_scaled[i]["uppBou"] - self.bounds_scaled[i]["lowBou"]))
        return initialValues

    def get_trimmed_df(self, filepath, aliases):
        """
        Create a dataFrame based on the given result-file and the aliases with the to_pandas_ebc function.
        :param filepath: str, os.path.normpath
        Path to the matfile
        :param aliases: dict
        Dictionary for selecting and naming the dataframe columns
        :return: df: pandas.DataFrame
        Data Frame object with relevant data
        """
        if filepath.endswith(".mat"):
            sim = sr.SimRes(filepath)
            df = sr_ebc.to_pandas_ebc(sim, names=list(aliases), aliases=aliases)
            return df
        else:
            raise TypeError("Given filename is not of type *.mat")

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

    def save_result(self, res):
        """
        Process the result, re-run the simualation and generate a logFile for the minimal quality measurement
        :param res: minimize.result
        Result object of the minimization
        :return:
        """
        print("Number of iterations: %s"%self.counter)
        print("Minimal %s: %s"%(self.qualMeas,res.fun))
        print("Initial Values for this minimum: %s"%(self._convSet(res.x)))

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
            if not ("meas" in goal and "sim" in goal and "weighting" in goal and len(goal.keys())==3):
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
        return True #If no error has occured, this check passes

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
            tempTunerPara[elem.attrib["name"]] = float(elem.text) #All values inside the dict are floats
        tunerPara[initalName.attrib["name"]] = tempTunerPara
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
            if elem.attrib["name"] == "weighting":
                goal[elem.attrib["name"]] = float(elem.text)
            else:
                goal[elem.attrib["name"]] = elem.text
        goals.append(goal)
    return goals

def save_tuner_xml(tunerPara, filepath):
    """
    Save a tuner param dictionary into a xml file
    :param tunerPara: dict
    Dictionary with information about tuner parameters. name, start-value, upper and lower-bound
    :param filepath: str, os.path.normpath
    :return: None
    """
    root = ET.Element("tunerParaDict")
    for initialName, iniNameDict in tunerPara.items():
        iniNameRoot = ET.SubElement(root, "initalName", name = initialName)
        for key, value in iniNameDict.items():
            ET.SubElement(iniNameRoot, key, name = key).text = str(value)
    _saveXML(filepath,root)

def save_goals_xml(goals, filepath):
    """
    Save a goals-list to a xml file
    :param goals: list
    List of dictionaries containing the goals
    :param filepath: str, os.path.normpath
    :return: None
    """
    root = ET.Element("goals")
    for goal in goals:
        goalRoot = ET.SubElement(root, "goal")
        for key, value in goal.items():
            ET.SubElement(goalRoot, key, name = key).text = str(value)
    _saveXML(filepath,root)

def _saveXML(filepath, root):
    """
    Saves a root in a readable str-xml file
    :param filepath: str, os.path.normpath
    :param root: xml-root
    :return: None
    """
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
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