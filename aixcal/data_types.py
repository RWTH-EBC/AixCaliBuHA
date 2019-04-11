import pandas as pd
import os
import modelicares.simres as sr


class TimeSeriesData(object):
    def __init__(self, filepath):
        self.data_type = None
        self.df = pd.DataFrame()
        # Check whether the file exists
        if not os.path.isfile(filepath): raise FileNotFoundError("The given filepath could not be openend")
        self.filepath = filepath
        self._load_data()

    def _load_data(self):
        """
        Private function to load the data in the file in filepath and convert it to a dataframe.
        """
        # Open based on file suffix. Currently, hdf, csv, and Modelica result files (mat) are supported.
        if self.filepath.endswith(".hdf"):
            self.df = pd.read_hdf(self.filepath)
        elif self.filepath.endswith(".csv"):
            self.df = pd.read_csv(self.filepath)
        elif self.filepath.endswith(".mat"):
            sim = sr.SimRes(self.filepath)
            self.df = sim.to_pandas()
        else:
            raise TypeError("Only .hdf and .csv are supported!")


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
    def __init__(self, name, initial_value, bounds = None):
        """
        Class for a tuner parameters.
        :param name: str
        Name of the tuner parameter
        :param initial_value: float, int
        Initial value for optimization
        :param bounds: list, tuple
        Tuple or list of float or ints for lower and upper bound to the tuner parameter
        """
        self.name = name
        self.initial_value = initial_value
        self.bounds = bounds
        self._assert_correct_input()

    def dump_to_dict(self):
        """Function to store the class in a dict to later save it to another format."""
        return self.__dict__

    def _assert_correct_input(self, ):
        """
        Function to check whether the class parameters are correct or not.
        This check is done to avoid errors at later stages of the optimization
        """
        if not isinstance(self.name, str): raise TypeError("Name has to be of type string")
        if not isinstance(self.initial_value, (float,int)): raise TypeError("Initial_value as to be of type float")
        if self.bounds:
            if not isinstance(self.bounds, (list, tuple)): raise TypeError("Bounds have to be a list or a tuple")
            if not len(self.bounds)==2:
                raise ValueError("The bounds object has to be of length 2 but has length %s"%len(self.bounds))
            if not isinstance(self.bounds[0], (float, int)) or not isinstance(self.bounds[1], (float, int)):
                raise TypeError("Given bounds are not of type float or int")
            if self.bounds[0] >= self.bounds[1]: raise ValueError("Lower bound is higher than upper bound")
            if self.initial_value < self.bounds[0] or self.initial_value > self.bounds[1]:
                raise ValueError("The initial value lays outside of the given boundaries")
        return True
