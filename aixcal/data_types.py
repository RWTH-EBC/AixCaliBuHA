import pandas as pd
import os
import modelicares.simres as sr


class TimeSeriesData(object):
    def __init__(self, filepath, **kwargs):
        """
        Base class for time series data in the framework.
        :param filepath: str, os.path.normpath
            Filepath ending with either .hdf, .mat or .csv containing
            time-dependent data to be loaded as a pandas.DataFrame
        :keyword key: Name of the table in a .hdf-file if the file
        contains multiple tables.
        """
        self.data_type = None
        self.df = pd.DataFrame()
        # Check whether the file exists
        if not os.path.isfile(filepath): raise FileNotFoundError("The given filepath (%s) could "
                                                                 "not be openend" % filepath)
        self.filepath = filepath
        # Used for import of .hdf-files, as multiple tables can be stored inside on file.
        if "key" in kwargs:
            self.key = kwargs["key"]
        self._load_data()


    def _load_data(self):
        """
        Private function to load the data in the file in filepath and convert it to a dataframe.
        """
        # Open based on file suffix. Currently, hdf, csv, and Modelica result files (mat) are supported.
        if self.filepath.endswith(".hdf"):
            self._load_hdf()
        elif self.filepath.endswith(".csv"):
            self.df = pd.read_csv(self.filepath)
        elif self.filepath.endswith(".mat"):
            sim = sr.SimRes(self.filepath)
            self.df = sim.to_pandas()
        else:
            raise TypeError("Only .hdf, .csv and .mat are supported!")

    def _load_hdf(self):
        """
        Load the current file as a hdf to a dataframe.
        As specifying the key can be a problem, the user will
        get all keys of the file if one is necessary but not provided.
        """
        try:
            self.df = pd.read_hdf(self.filepath, key=self.key)
        except (ValueError, KeyError):
            keys = ", ".join(get_keys_of_hdf_file(self.filepath))
            raise KeyError("key must be provided when HDF5 file contains multiple datasets. "
                           "Here are all keys in the given hdf-file: %s" % keys)


class MeasTargetData(TimeSeriesData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # Inherit all objects from TimeSeriesData
        self.data_type = "MeasTargetData"


class MeasInputData(TimeSeriesData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # Inherit all objects from TimeSeriesData
        self.data_type = "MeasInputData"


class SimTargetData(TimeSeriesData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # Inherit all objects from TimeSeriesData
        self.data_type = "SimTargetData"


class TunerPara(object):
    def __init__(self, name, initial_value, bounds=None):
        """
        Class for a tuner parameters.
        :param name: str
        Name of the tuner parameter
        :param initial_value: float, int
        Initial value for optimization
        :param bounds: list, tuple
        Tuple or list of float or ints for lower and upper bound to the tuner parameter
        """
        # TODO Decide on the format of the TunerPara-Class.
        #  Either on parameter per class, or a class with an array of parameters. Maybe a df is an option
        self.name = name
        self.initial_value = initial_value
        self.bounds = bounds
        self._assert_correct_input()
        self.scale = self.bounds[1] - self.bounds[0]
        self.offset = self.bounds[0]

    def scale(self, descaled):
        """
        Scales the given value to the bounds of the tuner parameter between 0 and 1
        :param descaled: float
        Value to be scaled
        :return: scaled: float
        Scaled value between 0 and 1
        """
        if not self.bounds:  # If no bounds are given, scaling is not possible--> descaled = scaled
            return descaled
        return (descaled - self.offset)/self.scale

    def descale(self, scaled):
        """
        Converts the given scaled value to an descaled one.
        :param scaled: float
        Scaled input value between 0 and 1
        :return: descaled: float
        descaled value based on bounds.
        """
        if not self.bounds:  # If no bounds are given, scaling is not possible--> descaled = scaled
            return scaled
        return scaled*self.scale + self.offset

    def dump_to_dict(self):
        """Function to store the class in a dict to later save it to another format."""
        return self.__dict__

    def _assert_correct_input(self, ):
        """
        Function to check whether the class parameters are correct or not.
        This check is done to avoid errors at later stages of the optimization.
        """
        if not isinstance(self.name, str): raise TypeError("Name has to be of type string")
        if not isinstance(self.initial_value, (float,int)): raise TypeError("Initial_value has to be of type float or int")
        if self.bounds:
            if not isinstance(self.bounds, (list, tuple)): raise TypeError("Bounds have to be a list or a tuple")
            if not len(self.bounds) == 2:
                raise ValueError("The bounds object has to be of length 2 but has length %s"%len(self.bounds))
            if not isinstance(self.bounds[0], (float, int)) or not isinstance(self.bounds[1], (float, int)):
                raise TypeError("Given bounds are not of type float or int")
            if self.bounds[0] >= self.bounds[1]: raise ValueError("Lower bound is higher than upper bound")
            if self.initial_value < self.bounds[0] or self.initial_value > self.bounds[1]:
                raise ValueError("The initial value lays outside of the given boundaries")
        return True


class Goal:
    def __init__(self, measured_data, simulated_data, weighting=1.0):
        """
        Class for one or multiple goals. Used to evaluate the difference between current simulation and
        :param measured_data: aixcal.data_types.MeasTargetData
            The dataset to be used as a reference for the simulation output.
        :param simulated_data: aixcal.data_types.SimTargetData
        :param weighting: float
            Value between 0 and 1 to account for multiple Goals to be evaluated.
        """

        self.measured_data = measured_data
        self.weighting = weighting
        self.simulated_data = simulated_data


class CalibrationClass:
    def __init__(self, name, start_time, stop_time, goals=None, tuner_para=None):
        """
        Class used for continuous calibration.
        :param name: str
        Name of the class, e.g. 'device on'
        :param start_time: float, int
        Time at which the class starts
        :param stop_time:
        Time at which the class ends
        :param goals: aixcal.data_types.Goal
        Goal parameters which are relevant in this class.
        As this class may be used in the classifier, a Goal-Class
        may not be available at all times and can be added later.
        :param tuner_para: aixcal.data_types.TunerPara
        As this class may be used in the classifier, a Tunerpara-Class
        may not be available at all times and can be added later.

        """
        self.name = name
        self.start_time = start_time
        self.stop_time = stop_time
        if goals:
            self.goals = self.set_goals(goals)
        if tuner_para:
            self.tuner_para = self.set_tuner_para(tuner_para)

    def set_goals(self, goals):
        if not isinstance(goals, type(Goal)):
            raise TypeError("Provided goals parameter in not of type Goal")
        self.goals = goals

    def set_tuner_para(self, tuner_para):
        if not isinstance(tuner_para, type(TunerPara)):
            raise TypeError("Given tuner_para is not of type TunerPara")
        self.tuner_para = tuner_para


def get_keys_of_hdf_file(filepath):
    """
    Find all keys in a given hdf-file.
    :param filepath: str, os.path.normpath
        Path to the .hdf-file
    :return: list
        List with all keys in the given file.
    """
    import h5py
    f = h5py.File(filepath, 'r')
    return list(f.keys())
