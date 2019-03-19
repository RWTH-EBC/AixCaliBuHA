"""
EITHER
Script with the dymola-interface class.
Create the object dymola-interface to simulate models.
"""

import os,sys, psutil
sys.path.insert(0, os.path.join('C:\Program Files\Dymola 2019',
                    'Modelica',
                    'Library',
                    'python_interface',
                    'dymola.egg'))

from dymola.dymola_interface import DymolaInterface

class DymolaAPI():
    def __init__(self, cwdir, packages, model_name):
        """
        Dymola interface class
        :param cwdir: str, os.path.normpath
        Dirpath for the current working directory of dymola
        :param packages: list
        List with path's to the packages needed to simulate the model
        :param model_name: str
        Name of the model to be simulated
        """
        self.cwdir = cwdir
        self.packages = packages
        self.model_name = model_name
        self.sim_setup = {'startTime': 0.0,
                         'stopTime':1.0,
                         'numberOfIntervals': 0,
                         'outputInterval':1,
                         'method': 'Dassl',
                         'tolerance': 0.0001,
                         'fixedstepsize': 0.0,
                         'resultFile': 'resultFile',
                         'autoLoad': None,
                         'initialNames':[],
                         'initialValues':[]}
        self.struc_params = []
        self.crit_number_instances = 10
        self._setup_dym()

    def simulate(self, save_files = True, save_name ="", get_structurals = False):
        """
        Simulate the current setup.
        If simulation terminates without an error and the files should be saved, the files are moved to a folder based on the current datetime.
        Returns the filepath of the result-matfile.
        :param save_files: bool
        True if the simulation results should be saved
        :param save_name: str
        Name of the folder inside the cwdir where the files will be saved
        :param get_structurals: bool
        True if structural parameters should be altered by using modifiers
        :return:
        """
        if self.struc_params:
            print("Warning: Currently, the model is retranslating for each simulation.\n"
                  "Check for these parameters: %s" %",".join(self.struc_params))
            self.model_name = self._alter_model_name(self.sim_setup, self.model_name, self.struc_params) #Alter the model_name for the next simulation
        res = self.dymola.simulateExtendedModel(self.model_name,
                                                startTime=self.sim_setup['startTime'],
                                                stopTime=self.sim_setup['stopTime'],
                                                numberOfIntervals=self.sim_setup['numberOfIntervals'],
                                                outputInterval=self.sim_setup['outputInterval'],
                                                method=self.sim_setup['method'],
                                                tolerance=self.sim_setup['tolerance'],
                                                fixedstepsize=self.sim_setup['fixedstepsize'],
                                                resultFile=self.sim_setup['resultFile'],
                                                initialNames=self.sim_setup['initialNames'],
                                                initialValues=self.sim_setup['initialValues'])
        if not res[0]:
            print("Simulation failed!")
            print("The last error log from Dymola:")
            print(self.dymola.getLastErrorLog())
            raise Exception("Simulation failed: Look into dslog.txt of working directory.")
        if save_files:
            new_path = os.path.join(self.cwdir, save_name) # create a new path based on the current datetime
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            for filename in ["%s.mat"%self.sim_setup["resultFile"], "dslog.txt", "dsfinal.txt"]:
                if os.path.isfile(os.path.join(new_path, filename)):
                    os.remove(os.path.join(new_path, filename)) #Delete existing files
                os.rename(os.path.join(self.cwdir, filename), os.path.join(new_path, filename)) #Move files
        else:
            new_path = self.cwdir
        if get_structurals:
            struc_params = self._filter_error_log(self.dymola.getLastErrorLog()) #Get the structural parameters based on the error log
            return True, os.path.join(new_path, "%s.mat" % self.sim_setup['resultFile']), struc_params
        else:
            return True, os.path.join(new_path, "%s.mat" % self.sim_setup['resultFile'])

    def set_initial_values(self, initial_values):
        """
        Overwrite inital values
        :param initial_values: list
        List containing initial values for the dymola interface
        :return:
        """
        self.sim_setup["initialValues"] = initial_values

    def set_sim_setup(self, sim_setup):
        """
        Overwrites multiple entries in the simulation setup dictionary
        :param sim_setup: dict
        Dictionary object with the same keys as this class's sim_setup dictionary
        :return:
        """
        for key, value in sim_setup.items():
            if not key in self.sim_setup:
                raise KeyError("The given simulation setup dictionary contains keys which are not usable in the dymola interface")
            else:
                if type(value) != type(self.sim_setup[key]):
                    raise TypeError("The given type is not valid for the dymola interface")
                else:
                    self.sim_setup[key] = value

    def import_initial(self, filepath):
        """
        Load given dsfinal.txt into dymola
        :param filepath: str, os.path.normpath
        Path to the dsfinal.txt to be loaded
        """
        assert os.path.isfile(filepath) , "Given filepath %s does not exist"%filepath
        assert ".txt" == os.path.splitext(filepath)[1], 'File is not of type .txt'
        res = self.dymola.importInitial(dsName=filepath)
        if res:
            print("\nSuccessfully loaded dsfinal.txt")
        else:
            raise Exception("Could not load dsfinal into Dymola.")

    def set_cwdir(self, cwdir):
        """Set the working directory to """
        self.cwdir = cwdir
        if hasattr(self, "dymola"):
            dymPath = self._make_dym_path(self.cwdir)
            res = self.dymola.cd(dymPath)
            if not res:
                raise OSError("Could not change working directorie to {}".format(self.cwdir))

    def _make_dym_path(self, path):
        """
        Convert given path to a path readable in dymola. If the path does not exist, create it.
        :param path: os.path.normpath, str
        :return: str
        Path readable in dymola
        """
        if not os.path.isdir(path):
            os.makedirs(path)
        path = path.replace("\\", "/")
        #Search for e.g. "D:testzone" and replace it with D:/testzone
        loc = path.find(":")
        if path[loc+1] != "/" and loc != -1:
            path = path.replace(":",":/")

        return path

    def _setup_dym(self):
        """Load all packages and change the current working directory"""
        self.dymola = DymolaInterface()
        self._check_dymola_instances()
        self.dymola.cd(self.cwdir)
        for pack in self.packages:
            print("Loading Model %s" % os.path.dirname(pack).split("\\")[-1])
            res = self.dymola.openModel(pack, changeDirectory=False)
            if not res:
                print(self.dymola.getLastErrorLog())
        print("Loaded modules")

    def _check_dymola_instances(self):
        counter = 0
        for proc in psutil.process_iter():
            try:
                if "Dymola" in proc.name():
                    counter += 1
            except psutil.AccessDenied:
                continue
        if counter >= self.crit_number_instances:
            print("WARNING: There are currently %s Dymola-Instances running on your machine!!!"%counter)
        return counter

    def _filter_error_log(self, error_log):
        """
        Filters the error log to detect reoccuring errors or structural parameters.
        Each structural parameter will raise this warning:
        'Warning: Setting n has no effect in model.\n
        After translation you can only set literal start-values and non-evaluated parameters.'
        Filtering of this string will extract 'n' in the given case.
        :param error_log: str
        Error log from the dymola_interface.getLastErrorLog() function
        :return: filtered_log: str
        """
        struc_params = []
        split_error = error_log.split("\n")
        for i in range(1, len(split_error)): #First line will never match the string
            if "After translation you can only set literal start-values and non-evaluated parameters" in split_error[i]:
                prev_line = split_error[i-1]
                param = prev_line.replace("Warning: Setting ", "").replace(" has no effect in model.","") #Obviously not the best way... but it works
                struc_params.append(param)
        return struc_params

    def _alter_model_name(self, sim_setup, model_name, struc_params):
        """
        Creates a modifier for all structural parameters, based on the modelname and the initalNames and values.
        :param sim_setup: dict
        Simulation setup dictionary
        :param model_name: str
        Name of the model to be modified
        :param struc_params: list
        List of strings with structural parameters
        :return: altered_modelName: str
        modified model name
        """
        initial_values = sim_setup["initialValues"]
        initial_names = sim_setup["initialNames"]
        model_name = model_name.split("(")[0] #Trim old modifier
        if struc_params == [] or initial_names == []:
            return
        all_modifiers = []
        for struc_para in struc_params:
            if struc_para in initial_names: #Checks if the structural parameter is inside the initialNames to be altered
                for k in range(0,len(initial_names)): #Get the location of the parameter for extraction of the corresponding initial value
                    if initial_names[k]==struc_para:
                        break
                all_modifiers.append("%s = %s"%(struc_para, initial_values[k]))
        altered_model_name = "%s(%s)"%(model_name, ",".join(all_modifiers))
        return altered_model_name
