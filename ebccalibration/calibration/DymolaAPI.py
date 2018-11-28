"""
EITHER
Script with the dymola-interface class.
Create the object dymola-interface to simulate models.
"""

import os,sys
sys.path.insert(0, os.path.join('C:\Program Files (x86)\Dymola 2018',
                    'Modelica',
                    'Library',
                    'python_interface',
                    'dymola.egg'))

from dymola.dymola_interface import DymolaInterface

class dymolaInterface():
    def __init__(self, cwdir, packages, modelName):
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

    def simulate(self, saveFiles = True, saveName = ""):
        """Simulate the current setup.
        If simulation terminates without an error and the files should be saved, the files are moved to a folder based on the current datetime.
        Returns the filepath of the result-matfile.
        Parameters:
        saveFiles    """
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
            for filename in ["%s.mat"%self.simSetup["resultFile"], "dslog.txt"]:
                if os.path.isfile(os.path.join(new_path, filename)):
                    os.remove(os.path.join(new_path, filename)) #Delete existing files
                os.rename(os.path.join(self.cwdir, filename), os.path.join(new_path, filename)) #Move files
        else:
            new_path = self.cwdir
        return True, os.path.join(new_path, "%s.mat"%self.simSetup['resultFile'])

    def set_startTime(self, startTime):
        """
        Set's the start-time of the simulation
        :param startTime:
        Start-time of experiment
        :return:
        """
        self.simSetup["startTime"] = startTime

    def set_endTime(self, stopTime):
        """
        Set's the stop-time of the simulation
        :param stopTime: float
        Stop time of experiment
        :return:
        """
        self.simSetup["stopTime"] = stopTime

    def set_initialNames(self, initialNames):
        """
        Overwrite inital names
        :param initialNames: list
        List containing initial names for the dymola interface
        :return:
        """
        self.simSetup["initialNames"] = initialNames
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

    def _setupDym(self):
        """Load all packages and change the current working directory"""
        self.dymola = DymolaInterface()
        self.dymola.cd(self.cwdir)
        for pack in self.packages:
            print("Loading Model %s" % os.path.dirname(pack).split("\\")[-1])
            res = self.dymola.openModel(pack, changeDirectory=False)
            if not res:
                print(self.dymola.getLastErrorLog())
        print("Loaded modules")