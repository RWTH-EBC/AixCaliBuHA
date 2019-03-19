import pandas as pd
import os

class TimeSeriesData(object):
    def __init__(self, filepath):
        self.data_type = None
        self.df = pd.DataFrame()
        if self._check_filepath():
            self.filepath = filepath
            self.load_data()
        #raise NotImplementedError

    def load_data(self):
        if self.filepath.endswith(".hdf"):
            self.df = pd.read_hdf(self.filepath)
        elif self.filepath.endswith(".csv"):
            self.df = pd.read_csv(self.filepath)
        else:
            raise TypeError("Only .hdf and .csv are supported!")

    def _check_filepath(filepath)
        if not os.path.isfile(filepath): return False

class MeasTargetData(TimeSeriesData):
    def __init__(self, filepath):
        super().__init__(filepath) # Inherit all objects from TimeSeriesData
        self.data_type = "MeasTargetData"

class MeasInputData(TimeSeriesData):
    def __init__(self, filepath):
        super().__init__(filepath)
        self.data_type = "MeasInputData"

class SimTargetData(TimeSeriesData):
    def __init__(self, filepath):
        super().__init__(filepath)
        self.data_type = "SimTargetData"

class TunerPara(object):
    def __init__(self, name, initial_value, bounds = None):
        self._assert_correct_input(name, initial_value, bounds)
        self.name = name
        self.initial_value = initial_value
        self.bounds = bounds

    def _assert_correct_input(self, name, initial_value, bounds):
        """
        Function to check whether the class paramters are correct or not.
        :param name: Name of the tuner parameter
        :param initial_value: Initial value
        :param bounds: Tuple for lower and upper bound to the value
        :return True if check is correct.
        """
        if not isinstance(name,str): raise TypeError("name has to be of type string")
        if not isinstance(initial_value, (float,int)): raise TypeError("initial_value as to be of type float")
        if bounds:
            if not len(bounds)==2:
                raise ValueError("The bounds object has to be of length 2 but has length%s"%len(bounds))
            if not isinstance(bounds[0], (float, int)) or not isinstance(bounds[1], (float, int)):
                raise TypeError("Given bounds are not of type float or int!")
            if bounds[0] >= bounds[1]: raise ValueError("Lower bound is higher than upper bound!")
            if initial_value < bounds[0] or initial_value > bounds[1]:
                raise ValueError("The initial value lays outside of the given boundaries!")
        return True