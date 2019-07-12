"""Base-module for the segmentizer-package.
"""

from abc import abstractmethod
import os
from aixcal.utils import visualizer


class Classifier:
    """
    Base-Class for a classifier. All classifiers should
    be able to process some MeasTarget-Data and MeasInput-Data
    into aixcal.data_types.CalibrationClass(es).

    :param str,os.path.normpath cd:
        Working directory for storing logs and plots
    """

    def __init__(self, cd, **kwargs):
        """Instantiate instance parameters"""
        #%%Set class parameters
        if not os.path.isdir(cd):
            os.mkdir(cd)
        self.cd = cd
        self.logger = visualizer.ClassifierVisualizer(cd, self.__class__.__name__)

        #%% Update kwargs:
        self.__dict__.update(kwargs)

    @abstractmethod
    def classify(self, df, **kwargs):
        """
        Base function for executing the classification based on the given data.

        :param pd.DataFrame df:
            Given dataframe may be extracted from the TimeSeriesData class. Should
            contain all relevant keys.
        :return: list
            List containing aixcal.data_types.CalibrationClass objects
        """
        raise NotImplementedError('{}.classify function is not '
                                  'defined'.format(self.__class__.__name__))
