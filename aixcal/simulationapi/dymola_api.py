import os
import sys
#import psutil
import warnings
DymolaInterface = None  # Create dummy to later be used for global-import


class DymolaAPI:
    def __init__(self, cd, packages, model_name, dymola_interface_path=None, show_window=False):
        """
        Dymola interface class
        :param cd: str, os.path.normpath
        Dirpath for the current working directory of dymola
        :param packages: list
        List with path's to the packages needed to simulate the model
        :param model_name: str
        Name of the model to be simulated
        """
        # First import the dymola-interface
        if dymola_interface_path:
            assert dymola_interface_path.endswith(".egg"), "Please provide an .egg-file for the dymola-interface."
            if not os.path.isfile(dymola_interface_path):
                raise FileNotFoundError("Given dymola-interface could not be found.")
        else:
            dymola_interface_path = get_dymola_interface_path()
            if not dymola_interface_path:
                raise FileNotFoundError("Could not find a dymola-interface on your machine.")
        _global_import_dymola(dymola_interface_path)
        self.cd = cd
        self.packages = packages
        # TODO Rethink the model_name placement. Maybe as a parameter for self.simulate?!
        self.model_name = model_name
        # Default simulation setup
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
        self._structural_params = []
        # Parameter for raising a warning if to many dymola-instances are running
        self._critical_number_instances = 10
        self._setup_dymola_interface(show_window)

    def simulate(self, save_files=True, save_name="", get_structural_paras=False):
        """
        Simulate the current setup.
        If simulation terminates without an error and the files should be saved, the files are moved to a folder based on the current datetime.
        Returns the filepath of the result-matfile.
        :param save_files: bool
        True if the simulation results should be saved
        :param save_name: str
        Name of the folder inside the cd where the files will be saved
        :param get_structural_paras: bool
        True if structural parameters should be altered by using modifiers
        :return:
        """
        if self._structural_params:
            print("Warning: Currently, the model is re-translating for each simulation.\n"
                  "Check for these parameters: %s" %",".join(self._structural_params))
            self.model_name = self._alter_model_name(self.sim_setup, self.model_name, self._structural_params) #Alter the model_name for the next simulation
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
        # TODO: Check the way the files are stored. Should this be part of the class?
        if save_files:
            new_path = os.path.join(self.cd, save_name) # create a new path based on the current datetime
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            for filepath in ["%s.mat"%self.sim_setup["resultFile"], "dslog.txt", "dsfinal.txt"]:
                if os.path.isfile(os.path.join(new_path, filepath)):
                    os.remove(os.path.join(new_path, filepath)) #Delete existing files
                os.rename(os.path.join(self.cd, filepath), os.path.join(new_path, filepath)) #Move files
        else:
            new_path = self.cd
        # TODO Where to put the analysis of the structural parameters?
        if get_structural_paras:
            structural_params = _filter_error_log(self.dymola.getLastErrorLog()) #Get the structural parameters based on the error log
            return True, os.path.join(new_path, "%s.mat" % self.sim_setup['resultFile']), structural_params
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
                raise KeyError("The given simulation setup dictionary contains "
                               "keys which are not usable in the dymola interface")
            else:
                if not isinstance(value, type(self.sim_setup[key])):
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

    def set_cd(self, cd):
        """Set the working directory to the given path"""
        # Check if dymola is already running
        if not hasattr(self, "dymola"):
            return

        dymola_path = _make_dym_path(cd)
        res = self.dymola.cd(dymola_path)
        if not res:
            raise OSError("Could not change working directory to {}".format(cd))
        else:
            self.cd = cd

    def _setup_dymola_interface(self, show_window):
        """Load all packages and change the current working directory"""
        self.dymola = DymolaInterface(showwindow=show_window)
        self._check_dymola_instances()
        self.set_cd(self.cd)
        for package in self.packages:
            print("Loading Model %s" % os.path.dirname(package).split("\\")[-1])
            res = self.dymola.openModel(package, changeDirectory=False)
            if not res:
                raise ImportError(self.dymola.getLastErrorLog())
        print("Loaded modules")

    def _check_dymola_instances(self):
        """
        Check how many dymola instances are running on the machine.
        Raise a warning if the number exceeds a certain amount.
        """
        counter = 0
        for proc in psutil.process_iter():
            try:
                if "Dymola" in proc.name():
                    counter += 1
            except psutil.AccessDenied:
                continue
        if counter >= self._critical_number_instances:
            warnings.warn("There are currently %s Dymola-Instances running on your machine!" % counter)

    def _alter_model_name(self, sim_setup, model_name, structural_params):
        """
        Creates a modifier for all structural parameters, based on the modelname and the initalNames and values.
        :param sim_setup: dict
        Simulation setup dictionary
        :param model_name: str
        Name of the model to be modified
        :param structural_params: list
        List of strings with structural parameters
        :return: altered_modelName: str
        modified model name
        """
        initial_values = sim_setup["initialValues"]
        initial_names = sim_setup["initialNames"]
        model_name = model_name.split("(")[0] # Trim old modifier
        if structural_params == [] or initial_names == []:
            return
        all_modifiers = []
        for structural_para in structural_params:
            # Checks if the structural parameter is inside the initialNames to be altered
            if structural_para in initial_names:
                # Get the location of the parameter for extraction of the corresponding initial value
                for k in range(0, len(initial_names)):
                    if initial_names[k] == structural_para:
                        break
                all_modifiers.append("%s = %s" % (structural_para, initial_values[k]))
        altered_model_name = "%s(%s)" % (model_name, ",".join(all_modifiers))
        return altered_model_name

# Stativ methods used in the DymolaAPI-class but may also be used by other modules.


def get_dymola_interface_path():
    """
    Function to get the path of the newest dymola interface
    installment on the used machine
    :return: str
    Path to the dymola.egg-file
    """
    path_to_egg_file = r"\Modelica\Library\python_interface\dymola.egg"
    syspaths = [r"C:\Program Files"]
    # Check if 64bit is installed
    systempath_64 = r"C:\Program Files (x86)"
    if os.path.isdir(systempath_64):
        syspaths.append(systempath_64)
    # Get all folders in both path's
    temp_list = []
    for systempath in syspaths:
        temp_list += os.listdir(systempath)
    # Filter programms that are not Dymola
    dym_versions = []
    for folder_name in temp_list:
        if "Dymola" in folder_name:
            dym_versions.append(folder_name)
    del temp_list
    # Find the newest version and return the egg-file
    dym_versions.sort()
    for dym_version in reversed(dym_versions):
        for system_path in syspaths:
            full_path = system_path+"\\"+dym_version+path_to_egg_file
            if os.path.isfile(full_path):
                return full_path
    # If still inside the function, no interface was found
    return None


def _global_import_dymola(dymola_interface_path):
    sys.path.insert(0, dymola_interface_path)
    global DymolaInterface
    try:
        from dymola.dymola_interface import DymolaInterface
    except ImportError:
        raise ImportError("Given dymola-interface could not be loaded:\n %s" % dymola_interface_path)


def _make_dym_path(path):
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


def _filter_error_log(error_log):
    """
    Filters the error log to detect recurring errors or structural parameters.
    Each structural parameter will raise this warning:
    'Warning: Setting n has no effect in model.\n
    After translation you can only set literal start-values and non-evaluated parameters.'
    Filtering of this string will extract 'n' in the given case.
    :param error_log: str
    Error log from the dymola_interface.getLastErrorLog() function
    :return: filtered_log: str
    """
    # TODO Check if regex could improve the filtering of the error log
    structural_params = []
    split_error = error_log.split("\n")
    for i in range(1, len(split_error)): #First line will never match the string
        if "After translation you can only set literal start-values and non-evaluated parameters" in split_error[i]:
            prev_line = split_error[i-1]
            param = prev_line.replace("Warning: Setting ", "").replace(" has no effect in model.", "")
            structural_params.append(param)
    return structural_params


if __name__ == "__main__":
    dymAPI = DymolaAPI(r"C:\\", [], "Test", show_window=True)
