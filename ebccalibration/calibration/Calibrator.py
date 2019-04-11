"""
Script for setting up the Calibrator-class. This class contains the goal-function, the
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

class Calibrator:
    def __init__(self, goals, tuner_para, qual_meas, method, dymola_api, bounds = None, time_info_tupel = None, **kwargs):
        """
        Class for a Calibrator.
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
        :param tuner_para: dict
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
        :param qual_meas: str
        Statistical value for the objective function, e.g. RMSE, MAE etc.
        :param method: str
        Used method for minimizer. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html for more info
        :param dymola_api: DymolaAPI
        Class for executing a simulation in dymola
        :param bounds: optimize.Bounds object, Default: None
        Used if the boundaries differ from 0 and 1 (already scaled / normalized)
        :param time_info_tupel: tupel
        Tupel with start_time and entTime info special sections of the resulting array are of interest for the objective function.
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
        if self._check_goals(goals):
            self.goals = goals
        if self._check_tuner_paras(tuner_para):
            self.tuner_para = tuner_para
        self.inital_set = [] #Create inital set for first iteration
        self.bounds_scaled = [] #Create a list to denormalize the sets for the simulation
        for key, value in tuner_para.items():
            self.bounds_scaled.append({"uppBou": value["uppBou"],
                                       "lowBou": value["lowBou"]})
            self.inital_set.append((value["start"] - value["lowBou"]) / (value["uppBou"] - value["lowBou"]))
        if not bounds:
            self.bounds = opt.Bounds(np.zeros(len(self.inital_set)), np.ones(len(self.inital_set))) #Define boundaries for normalized values, e.g. [0,1]
        #Set other values
        self.method = method
        self.qual_meas = qual_meas
        #Create bounds based on the dictionary.
        self.dymola_api = dymola_api
        #Create aliases dict out of given goals-dict.
        self.aliases = {}
        for goal in goals:
            self.aliases[goal["sim_full_modelica_name"]] = goal["sim"]
            self.aliases[goal["meas_full_modelica_name"]] = goal["meas"]
        self.dymola_api.sim_setup["initialNames"] = list(self.tuner_para)
        #Starttime and endtime
        assert len(time_info_tupel) == 2, "The lenght of the time_info_tupel has to be equal to two. First entry the starttime, second entry the endtime"
        self.time_info_tupel = time_info_tupel
        if self.time_info_tupel:
            self.start_time = self.time_info_tupel[0]
            self.end_time = self.time_info_tupel[1]

        #kwargs
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)
        if not hasattr(self, "method_options"):
            self.method_options = {}
        if not hasattr(self, "tol"):
            self.tol = None
        boolean_kwargs = ["plot_callback", "save_files", "continouus_calibration"]
        for boolean in boolean_kwargs:
            if not hasattr(self, boolean):
                setattr(self, boolean, False)
        #Set counter for number of iterations of the objective function
        self.counter = 0
        self.obj_his = []
        self.counter_his = []
        self.last_set = np.array([]) # Used if simulation failes
        self.log = ""
        if self.continouus_calibration:
            self.total_min = {"obj": 1e10,
                             "dsfinal":"",
                             "dsres":""}
            self.savepath_min_result = os.path.join(self.dymola_api.cwdir, "total_min")
            if not os.path.isdir(self.dymola_api.cwdir):
                print("Creating working directory {}".format(self.dymola_api.cwdir))
                os.mkdir(self.dymola_api.cwdir)
            if not os.path.isdir(self.savepath_min_result):
                os.mkdir(self.savepath_min_result)

    def calibrate(self, obj):
        """
        Optimizes the given inital set of parameters with the objective function.
        :param obj: callable
        Objective function for the minimize function
        :return: OptimizeResult
        Result of the optimization. Most important: x and fun
        """
        info_string = self._get_name_info_string()
        print(info_string)
        self.log += "\n" + info_string
        try:
            res = opt.minimize(fun=obj, x0=np.array(self.inital_set), method=self.method, bounds=self.bounds, tol=self.tol, options=self.method_options)
            if not self.continouus_calibration:
                self.dymola_api.dymola.close() # Close dymola
            self.save_log_as_csv()
            return res
        except Exception as e:
            print("Parameter set which caused failed simulation:")
            print(self._get_val_info_string(self.last_set, for_log= True))
            self.dymola_api.dymola.close()
            raise Exception(e) # Actually raise the error so the script stops.

    def cal_dlib(self, obj):
        import dlib
        res = dlib.find_min_global()

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
        conv_set = self._conv_set(set) #Convert set if multiple goals of different scales are used
        self.dymola_api.set_initial_values(conv_set) #Set initial values
        save_name = "%s"%str(self.counter) #Generate the folder name for the calibration
        success, filepath, struc_params = self.dymola_api.simulate(save_files=self.save_files, save_name=save_name, get_structurals=True) #Simulate
        if success:
            self.dymola_api.struc_params = struc_params
            df = self.get_trimmed_df(filepath, self.aliases, getTrajectorieNames = self.continouus_calibration) #Get results
            if self.time_info_tupel:
                # Trim results based on start and endtime
                df = df[self.start_time:self.end_time]
        total_res = 0
        for goal in self.goals:
            goal_res = self.calc_stat_values(df[goal["meas"]], df[goal["sim"]])
            total_res += goal["weighting"] * goal_res[self.qual_meas]
        self.obj_his.append(total_res)
        self.counter_his.append(self.counter)
        self.callbackF(set)
        if self.continouus_calibration and self.total_min["obj"] > total_res:
            self.total_min = {"obj":total_res,
                             "dsres": filepath,
                             "dsfinal":os.path.join(os.path.dirname(filepath), "dsfinal.txt")}
            #Overwrite old results:
            if os.path.isfile(os.path.join(self.savepath_min_result, "dsres.mat")):
                os.remove(os.path.join(self.savepath_min_result, "dsres.mat"))
            if os.path.isfile(os.path.join(self.savepath_min_result, "dsfinal.txt")):
                os.remove(os.path.join(self.savepath_min_result, "dsfinal.txt"))
            os.rename(filepath, os.path.join(self.savepath_min_result, "dsres.mat"))
            os.rename(os.path.join(os.path.dirname(filepath), "dsfinal.txt"), os.path.join(self.savepath_min_result, "dsfinal.txt"))
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
        info_string = self._get_val_info_string(xk)
        print(info_string)
        self.log += "\n" + info_string
        if self.plot_callback:
            if not hasattr(self, "fig"): # Instanciate the figure and ax
                self.fig, self.ax = plt.subplots(1, 1)
            self.ax.plot(self.counter_his[-1], self.obj_his[-1], "ro")
            self.ax.set_ylabel(self.qual_meas)
            self.ax.set_xlabel("Number iterations")
            self.ax.set_title(self.dymola_api.modelName)
            #If the changes are small, it seems like the plot does not fit the printed values. This boolean assures that no offset is used.
            self.ax.ticklabel_format(useOffset=False)
            plt.draw()
            plt.pause(1e-5)

    def _conv_set(self, set):
        """
        Convert given set to initial values in modelica according to function:
        iniVal_i = set_i*(max(iniVal_i)-min(iniVal_i)) + min(iniVal_i)
        :param set:
        Array with normalized values for the minimizer
        :return:
        List of inital values for dymola
        """
        initial_values = []
        for i in range(0, len(set)):
            initial_values.append(set[i]*(self.bounds_scaled[i]["uppBou"] - self.bounds_scaled[i]["lowBou"]) + self.bounds_scaled[i]["lowBou"])
        return initial_values


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
            self.traj_names = sr_ebc.get_trajectories(sim)
        return df


    def calc_stat_values(self, meas, sim):
        """
        Calculates multiple statistical values for the given numpy array of measured and simulated data.
        Calculates:
        MAE(Mean absolute error), RMSE(root mean square error), R2(coefficient of determination), CVRMSE(variance of RMSE), NRMSE(Normalized RMSE)
        :param meas:
        Array with measurement data
        :param sim:
        Array with simulation data
        :return: stat_values: dict
        Containing all calculated statistical values
        """
        stat_values = {}
        stat_values["MAE"] = skmetrics.mean_absolute_error(meas, sim)
        stat_values["RMSE"] = np.sqrt(skmetrics.mean_squared_error(meas, sim))
        stat_values["R2"] = 1 - skmetrics.r2_score(meas, sim)
        if np.mean(meas) != 0:
            stat_values["CVRMSE"] = stat_values["RMSE"] / np.mean(meas)
        else:
            if self.qual_meas == "CVRMSE":
                raise ValueError(
                    "The experimental gathered data has a mean of zero over the given timeframe.The CVRMSE can not be calculated. Please use the NRMSE")
            else:
                stat_values["CVRMSE"] = 1e10  # Punish the division by zero
        if (np.max(meas) - np.min(meas)) != 0:
            stat_values["NRMSE"] = stat_values["RMSE"] / (np.max(meas) - np.min(meas))
        else:
            if self.qual_meas == "NRMSE":
                raise ValueError(
                    "The experimental gathered data is constant over the given timeframe. The NRMSE can not be calculated. Please use the CVRMSE")
            else:
                stat_values["NRMSE"] = 1e10  # Punish the division by zero
        return stat_values

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
        result_log = "Results for calibration of model: %s\n"%self.dymola_api.modelName
        result_log += "Minimal %s: %s\n"%(self.qual_meas, res.fun)
        result_log += "Final parameter values:\n"
        result_log += "%s\n"%self._get_name_info_string(for_log=True)
        result_log += "%s\n"%self._get_val_info_string(res.x, for_log=True)
        result_log += "Number of iterations: %s\n"%self.counter
        result_log += "\nIteration log:\n" + self.log
        datestring = datetime.now().strftime("%Y%m%d_%H%M%S")
        f = open(os.path.join(savepath, "CalibrationLog_%s.txt"%datestring), "a+")
        f.write(result_log)
        f.close()
        if self.plot_callback:
            plt.savefig(os.path.join(savepath, "iterationPlot.%s"%ftype))

    def save_log_as_csv(self, sep = ";"):
        """
        Saves the log-string as a csv-file
        :param sep: str
        Seperator used in csv
        """
        savepath = os.path.join(self.dymola_api.cwdir, "log.csv")
        lines = self.log.split("\n")
        lines_csv = [re.sub("\s+", sep, line.strip()) for line in lines]
        f = open(savepath, "a+")
        f.seek(0)
        f.truncate()
        f.write("\n".join(lines_csv[1:]))
        f.close()

    def _check_goals(self, goals):
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

    def _check_tuner_paras(self, tuner_para):
        """
        Checks given tuner-para dictionary on correct formatting
        :param tuner_para:
        Dictionary with information about tuner parameters. name, start-value, upper and lower-bound
        :return: bool: True on success
        """
        if type(tuner_para) != type({}) or len(tuner_para) == 0: #Has to be type list and contain at least one entry
            raise TypeError("Given tuner parameters dictionary is not of type dict or is empty")
        for key, para_dict in tuner_para.items():
            if type(key) != type(""):
                raise TypeError("Parameter name %s is not of type string"%str(key))
            if not ("start" in para_dict and "uppBou" in para_dict and "lowBou" in para_dict and len(para_dict.keys())==3):
                raise Exception("Given tuner_para dict %s is no well formatted."%para_dict)
            if para_dict["uppBou"] - para_dict["lowBou"] <= 0:
                raise ValueError("The given upper boundary is less or equal to the lower boundary.")
        return True #If no error has occured, this check passes

    def _get_name_info_string(self, for_log = False):
        """
        Returns a string with the names of current tunerParameters
        :param for_log: Boolean
        If the string is created for the final log file, the best obj is not of interest
        :return: str
        The desired string
        """
        initial_names = list(self.tuner_para.keys())
        if not for_log:
            info_string = "{0:4s}".format("Iter")
        else:
            info_string = ""
        for i in range(0, len(initial_names)):
            info_string += "   {0:9s}".format(initial_names[i])

        if not for_log:
            info_string += "   {0:9s}".format(self.qual_meas)
        else:
            info_string = info_string[3:]
        return info_string

    def _get_val_info_string(self, set, for_log = False):
        """
        Returns a string with the values of current tunerParameters
        :param set: np.array
        Array with the current values of the calibration
        :param for_log: Boolean
        If the string is created for the final log file, the best obj is not of interest
        :return: str
        The desired string
        """
        ini_vals = self._conv_set(set)
        if not for_log:
            info_string = '{0:4d}'.format(self.counter)
        else:
            info_string = ""
        for i in range(0, len(ini_vals)):
            info_string += "   {0:3.6f}".format(ini_vals[i])
        # Add the last return value of the objective function.
        if not for_log:
            info_string += "   {0:3.6f}".format(self.obj_his[-1])
        else:
            info_string = info_string[3:]
        return info_string

###General functions:
def load_tuner_xml(filepath):
    """
    Load the tuner parameters dictionary from the given xml-file
    :param filepath: str, os.path.normpath
    :return: tuner_para: dict
    Dictionary with information about tuner parameters. name, start-value, upper and lower-bound
    """
    root = _load_xml(filepath)
    tuner_para = {}
    for inital_name in root:
        temp_tuner_para = {}
        for elem in inital_name:
            temp_tuner_para[elem.tag] = float(elem.text) #All values inside the dict are floats
        tuner_para[inital_name.tag] = temp_tuner_para
    return tuner_para

def load_goals_xml(filepath):
    """
    Load the goals from the given xml-file
    :param filepath: str, os.path.normpath
    :return: goals: list
    List of dictionaries containing the goals
    """
    root = _load_xml(filepath)
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

def save_xml(filepath, object):
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

def _load_xml(filepath):
    """
    Loads any xml file
    :param filepath: str, os.path.normpath
    :return:
    root of the xml
    """
    tree = ET.parse(filepath)  # Get tree of XML-file
    return tree.getroot()  # The root holds all plot-elements

def join_tuner_paras(continouus_data):
    """
    Join all initialNames used for calibration in the given dataset.
    :param continouus_data: list
    Contains dictionaries with goals, tunerParas etc.
    :return: list
    Joined list
    """
    joined_list = []
    for c in continouus_data:
        for name in c["tuner_para"].keys():
            if name not in joined_list:
                joined_list.append(name)
    return joined_list

def alter_tuner_paras(new_tuner, cal_history):
    """
    Based on old calibration results, this function alters the start-values for the new tuner_para-Set
    :param new_tuner: dict
    Tuner Parameter dict for the next time-interval
    :param cal_history: list
    List with all results and data from previous calibrations
    :return: tunerParaDict: dict
    Dictionary with the altered tunerParas
    """
    new_start_dict = _get_continouus_averages(cal_history)
    for key, value in new_tuner.items():
        if key in new_start_dict: #Check if the parameter will be used or not.
            new_tuner[key]["start"] = new_start_dict[key]
    return new_tuner

def _get_continouus_averages(cal_history):
    """
    Function to get the average value of a tuner parameter based on time- average
    :param cal_history: list
    List with all results and data from previous calibrations
    :return: new_start_dict: dict
    Dictionary with the average start parameters
    """
    #Iterate over calibration historie and create a new dictionary with the start-values for a given initialName#
    new_start_dict = {}
    total_time = 0
    for cal_his in cal_history:
        res_ini_vals = cal_his["cal"]._conv_set(cal_his["res"].x)
        timedelta = cal_his["continouus_data"]["stop_time"] - cal_his["continouus_data"]["start_time"]
        total_time += timedelta
        tuner_para = cal_his["continouus_data"]["tuner_para"]
        for i in range(0,len(tuner_para.keys())):
            ini_name = list(tuner_para.keys())[i]
            if ini_name in new_start_dict:
                new_start_dict[ini_name] += res_ini_vals[i] * timedelta
            else:
                new_start_dict[ini_name] = res_ini_vals[i] * timedelta
    for key, value in new_start_dict.items():
        new_start_dict[key] = value/total_time #Build average again
    return new_start_dict