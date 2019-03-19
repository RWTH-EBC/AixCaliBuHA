import pandas as pd
import os

class TimeSeriesData():
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
