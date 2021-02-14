"""Base-module for the whole calibration pacakge.
Used to define Base-Classes such as Optimizer and
Calibrator."""

from abc import abstractmethod
from ebcpy import optimization
from aixcalibuha import Goals, TunerParas


class Calibrator(optimization.Optimizer):
    """Base class for calibration in aixcalibuha. All classes
    performing calibration tasks must inherit from this
    class.
    """

    tuner_paras = TunerParas
    goals = Goals

    def __init__(self, cd, sim_api, statistical_measure, **kwargs):
        super().__init__(cd, **kwargs)
        self.sim_api = sim_api
        self.statistical_measure = statistical_measure

    @abstractmethod
    def obj(self, xk, *args):
        """
        Base objective function for any kind of calibration. This function has to
        be overwritten.
        """
        raise NotImplementedError('{}.obj function is not defined'.format(self.__class__.__name__))

    @abstractmethod
    def calibrate(self, framework, method=None):
        """
        Base calibration function. The idea is to call this function and start the
        calibration-process. This function has to be overwritten.
        """
        raise NotImplementedError('{}.calibrate function is not'
                                  ' defined'.format(self.__class__.__name__))

    @abstractmethod
    def validate(self, goals):
        """
        Function to use different measurement data and run the objective function
        again to validate the calibration. The final parameter vector of the
        calibration is used.

        :param aixcalibuha.Goals goals:
            Goals with data to be validated
        """
        raise NotImplementedError('{}.validate function is not'
                                  ' defined'.format(self.__class__.__name__))
