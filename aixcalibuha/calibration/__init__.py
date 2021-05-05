"""Base-module for the whole calibration package.
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
        raise NotImplementedError(f'{self.__class__.__name__}.obj function is not defined')

    @abstractmethod
    def calibrate(self, framework, method=None):
        """
        Base calibration function. The idea is to call this function and start the
        calibration-process. This function has to be overwritten.
        """
        raise NotImplementedError(f'{self.__class__.__name__}.calibrate function is not defined')

    @abstractmethod
    def validate(self, **kwargs):
        """
        Function to use different measurement data and run the objective function
        again to validate the calibration. The final parameter vector of the
        calibration is used.
        """
        raise NotImplementedError('{self.__class__.__name__}.validate function is not defined')
