"""
Script for setting up the calibrator-class. This class contains the goal-function, the
calculation of the statistical values and minimizer
"""
import scipy.optimize as opt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import ebcpython.modelica.postprocessing.simres_ebc as sr_ebc
from modelicares import simres as sr
from datetime import datetime #Used for saving of relevant files

class calibrator():
    def __init__(self, goals, tunerPara, qualMeas, method, dymAPI, bounds = None):
        """Class for a calibrator.
        Parameters:
        ---------------------
        goals: list
        Dictionary with information about the goal-variables. Each goal has a sub-dictionary.
            meas: str
            Name of measured data in dataframe
            sim: str
            Name of simulated data in dataframe
            weighting: float
            Weighting for the resulting objective function
        tunerPara: dict
        Dictionary with information about tuner parameters. name, start-value, upper and lower-bound
        qualMeas: str
        Statistical value for the objective function, e.g. RMSE, MAE etc.
        method: str
        Used method for minimizer. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html for more info
        dymAPI: dymolaInterface, dymolaShell
        Class for executing a simulation in dymola
        bounds: optimize.Bounds object, Default: None
        Used if the boundaries differ from 0 and 1
        """
        self.goals = goals
        self.tunerPara = tunerPara
        self.initalSet = [] #Create inital set for first iteration
        self.bounds_scaled = [] #Create a list to denormalize the sets for the simulation
        for key, value in tunerPara.items():
            self.bounds_scaled.append({"uppBou": value["uppBou"],
                                       "lowBou": value["lowBou"]})
            self.initalSet.append(value["start"]/value["uppBou"]-value["lowBou"])
        if not bounds:
            self.bounds = opt.Bounds(np.zeros(len(self.initalSet)),np.ones(len(self.initalSet))) #Define boundaries for normalized values, e.g. [0,1]
        self.methodOptions = {"disp":False,
                    "ftol":2.220446049250313e-09,
                    "eps":1e-8
                   }
        #Set other values
        self.method = method
        self.qualMeas = qualMeas
        #Create bounds based on the dictionary.
        self.dymAPI = dymAPI
        #Set counter for number of iterations of the objective function
        self.counter = 0
        self.startDateTime = datetime.now().strftime("%Y%m%d_%H%M%S")

    def calibrate(self, obj):
        """Optimizes the given inital set of parameters with the objective function.
        Parameters:
        --------------------
        obj: callable
        Objective function for the minimize function
        initalSet: list
        List with initial values for the minimize functions

        Returns:
        --------------------
        res: OptimizeResult
        Result of the optimization. Most important: x and fun"""
        res = opt.minimize(obj, np.array(self.initalSet), method=self.method, bounds= self.bounds, options=self.methodOptions, callback=callbackF)
        return res

    def objective(self, set):
        """Default objective function.
        The usual function will be implemented here:
        1. Convert the set to modelica-units
        2. Simulate the converted-set
        3. Get data as a dataFrame
        4. Calculate the objective based on statistical values"""
        self.counter += 1 #Increase counter
        conv_set = self.convSet(set) #Convert set if multiple goals of different scales are used
        self.dymAPI.set_initialValues(conv_set) #Set initial values
        saveName = "%s_%s"%(self.startDateTime,str(self.counter)) #Generate the folder name for the calibration
        success, filepath = self.dymAPI.simulate(saveFiles=True, saveName=saveName) #Simulate
        if success:
            df = self.get_trimmed_df(filepath) #Get results
        else:
            return 1e10 #punish failure of simulation
        total_res = 0
        for goal in self.goals:
            goal_res = self.calc_statValues(df[goal["meas"]],df[goal["sim"]])
            total_res += goal["weighting"] * goal_res[self.qualMeas]
        return total_res

    def callbackF(self, set):

    def convSet(self, set):
        """Convert given set to initial values in modelica"""
        initialValues = []
        for i in range(0, len(set)):
            initialValues.append(set[i]*self.bounds_scaled[i]["uppBou"] - self.bounds_scaled[i]["lowBou"])
        return initialValues

    def get_trimmed_df(self, filepath, aliases):
        """Create a dataFrame based on the given result-file and the aliases with the to_pandas_ebc function."""
        if filepath.endswith(".mat"):
            sim = sr.SimRes(filepath)
            df = sr_ebc.to_pandas_ebc(sim, names=list(aliases), aliases=aliases)
            return df
        else:
            raise TypeError("Given filename is not of type *.mat")

    def calc_statValues(self, meas, sim):
        """Calculates multiple statistical values for the given numpy array of measured and simulated data.
        Calculates:
        MAE(Mean absolute error), RMSE(root mean square error), R2(coefficient of determination), CVRMSE(variance of RMSE), NRMSE(Normalized RMSE)"""
        statValues = {}
        statValues["MAE"] = mean_absolute_error(meas, sim)
        statValues["RMSE"] = np.sqrt(mean_squared_error(meas, sim))
        statValues["R2"] = 1 - r2_score(meas, sim)
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
        """Process the result, re-run the simualation and generate a logFile for the minimal quality measurement"""
        print(res.fun)
        print(res.x)


###General functions:
def load_tuner_xml():
    pass

def save_tuner_xml():
    pass

def load_goals_xml():
    pass

def save_goals_xml():
    pass