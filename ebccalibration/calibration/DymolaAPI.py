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

class dymolaInterface():
    def __init__(self, cwdir, packages, modelName):
        """
        Dymola interface class
        :param cwdir: str, os.path.normpath
        Dirpath for the current working directory of dymola
        :param packages: list
        List with path's to the packages needed to simulate the model
        :param modelName: str
        Name of the model to be simulated
        """
        self.cwdir = cwdir
        self.packages = packages
        self.modelName = modelName
        self.simSetup = {'startTime': 0.0,
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
        self.strucParams = []
        self.critNumberInstances = 10
        self._setupDym()

    def simulate(self, saveFiles = True, saveName = "", getStructurals = False):
        """
        Simulate the current setup.
        If simulation terminates without an error and the files should be saved, the files are moved to a folder based on the current datetime.
        Returns the filepath of the result-matfile.
        :param saveFiles: bool
        True if the simulation results should be saved
        :param saveName: str
        Name of the folder inside the cwdir where the files will be saved
        :param getStructurals: bool
        True if structural parameters should be altered by using modifiers
        :return:
        """
        if self.strucParams:
            print("Warning: Currently, the model is retranslating for each simulation.\n"
                  "Check for these parameters: %s"%",".join(self.strucParams))
            self.modelName = self._alterModelName(self.simSetup, self.modelName, self.strucParams) #Alter the modelName for the next simulation
        res = self.dymola.simulateExtendedModel(self.modelName,
                                                 startTime=self.simSetup['startTime'],
                                                 stopTime=self.simSetup['stopTime'],
                                                 numberOfIntervals=self.simSetup['numberOfIntervals'],
                                                 outputInterval=self.simSetup['outputInterval'],
                                                 method=self.simSetup['method'],
                                                 tolerance=self.simSetup['tolerance'],
                                                 fixedstepsize=self.simSetup['fixedstepsize'],
                                                 resultFile=self.simSetup['resultFile'],
                                                 initialNames=self.simSetup['initialNames'],
                                                 initialValues=self.simSetup['initialValues'])
        if not res[0]:
            print("Simulation failed!")
            print(self.dymola.getLastErrorLog())
            return False, None
        if saveFiles:
            new_path = os.path.join(self.cwdir, saveName) # create a new path based on the current datetime
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            for filename in ["%s.mat"%self.simSetup["resultFile"], "dslog.txt", "dsfinal.txt"]:
                if os.path.isfile(os.path.join(new_path, filename)):
                    os.remove(os.path.join(new_path, filename)) #Delete existing files
                os.rename(os.path.join(self.cwdir, filename), os.path.join(new_path, filename)) #Move files
        else:
            new_path = self.cwdir
        if getStructurals:
            strucParams = self._filterErrorLog(self.dymola.getLastErrorLog()) #Get the structural parameters based on the error log
            return True, os.path.join(new_path, "%s.mat"%self.simSetup['resultFile']), strucParams
        else:
            return True, os.path.join(new_path, "%s.mat"%self.simSetup['resultFile'])

    def set_initialValues(self, initialValues):
        """
        Overwrite inital values
        :param initialValues: list
        List containing initial values for the dymola interface
        :return:
        """
        self.simSetup["initialValues"] = initialValues

    def set_simSetup(self, simSetup):
        """
        Overwrites multiple entries in the simulation setup dictionary
        :param simSetup: dict
        Dictionary object with the same keys as this class's simSetup dictionary
        :return:
        """
        for key, value in simSetup.items():
            if not key in self.simSetup:
                raise KeyError("The given simulation setup dictionary contains keys which are not usable in the dymola interface")
            else:
                if type(value) != type(self.simSetup[key]):
                    raise TypeError("The given type is not valid for the dymola interface")
                else:
                    self.simSetup[key] = value

    def importInitial(self, filepath):
        """
        Load given dsfinal.txt into dymola
        :param filepath: str, os.path.normpath
        Path to the dsfinal.txt to be loaded
        """
        assert os.path.isfile(filepath) , "Given filepath %s does not exist"%filepath
        assert ".txt" == os.path.splitext(filepath)[1], 'File is not of type .txt'
        res = self.dymola.importInitial(dsName=filepath)
        if res:
            print("Successfully loaded dsfinal.txt")
        else:
            raise Exception("Could not load dsfinal into dymola.")

    def _setupDym(self):
        """Load all packages and change the current working directory"""
        self.dymola = DymolaInterface()
        self._checkDymolaInstances()
        self.dymola.cd(self.cwdir)
        for pack in self.packages:
            print("Loading Model %s" % os.path.dirname(pack).split("\\")[-1])
            res = self.dymola.openModel(pack, changeDirectory=False)
            if not res:
                print(self.dymola.getLastErrorLog())
        print("Loaded modules")

    def _checkDymolaInstances(self):
        counter = 0
        for proc in psutil.process_iter():
            try:
                if "Dymola" in proc.name():
                    counter += 1
            except psutil.AccessDenied:
                continue
        if counter >= self.critNumberInstances:
            print("WARNING: There are currently %s Dymola-Instances running on your machine!!!"%counter)
        return counter

    def _filterErrorLog(self, errorLog):
        """
        Filters the error log to detect reoccuring errors or structural parameters.
        Each structural parameter will raise this warning:
        'Warning: Setting n has no effect in model.\n
        After translation you can only set literal start-values and non-evaluated parameters.'
        Filtering of this string will extract 'n' in the given case.
        :param errorLog: str
        Error log from the dymola_interface.getLastErrorLog() function
        :return: filtered_log: str
        """
        structuralParams = []
        splitError = errorLog.split("\n")
        for i in range(1, len(splitError)): #First line will never match the string
            if "After translation you can only set literal start-values and non-evaluated parameters" in splitError[i]:
                prev_line = splitError[i-1]
                param = prev_line.replace("Warning: Setting ", "").replace(" has no effect in model.","") #Obviously not the best way... but it works
                structuralParams.append(param)
        return structuralParams

    def _alterModelName(self, simSetup, modelName, strucParams):
        """
        Creates a modifier for all structural parameters, based on the modelname and the initalNames and values.
        :param simSetup: dict
        Simulation setup dictionary
        :param modelName: str
        Name of the model to be modified
        :param strucParams: list
        List of strings with structural parameters
        :return: altered_modelName: str
        modified model name
        """
        iniVals = simSetup["initialValues"]
        iniNames = simSetup["initialNames"]
        modelName = modelName.split("(")[0] #Trim old modifier
        if strucParams == [] or iniNames == []:
            return
        all_modifiers = []
        for strucPara in strucParams:
            if strucPara in iniNames: #Checks if the structural parameter is inside the initialNames to be altered
                for k in range(0,len(iniNames)): #Get the location of the parameter for extraction of the corresponding initial value
                    if iniNames[k]==strucPara:
                        break
                all_modifiers.append("%s = %s"%(strucPara, iniVals[k]))
        alteredModelName = "%s(%s)"%(modelName, ",".join(all_modifiers))
        return alteredModelName
