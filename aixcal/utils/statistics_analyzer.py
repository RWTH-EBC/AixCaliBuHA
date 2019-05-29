"""Module for calculating statistical
measures based on given methods."""

import numpy as np
import sklearn.metrics as skmetrics


class StatisticsAnalyzer:
    """Class for calculation of the statistical measure based on the
    given method. Either instantiate the class and run
    StatisticsAnalyzer.calc(meas, sim), or go for direct calculation with
    StatisticsAnalyzer.calc_METHOD(meas, sim).
    :param method: str
        One of the following:
            - MAE(Mean absolute error)
            - R2(coefficient of determination)
            - MSE (Mean squared error)
            - RMSE(root mean square error)
            - CVRMSE(variance of RMSE)
            - NRMSE(Normalized RMSE)
    """

    _supported_methods = ["MAE", "R2", "MSE", "RMSE", "CVRMSE", "NRMSE"]

    def __init__(self, method):
        """Instantiate class parameters"""
        try:
            exec("self.calc = self.calc_{}".format(method))
        except AttributeError:
            raise ValueError("The given method {} is not supported.\n"
                             "Choose one out of: {}".format(method, ", ".join(self._supported_methods)))

    @staticmethod
    def calc(meas, sim):
        """Placeholder class before instantiating the class correctly."""
        raise NotImplementedError('Instantiate the class to call this function.\n'
                                  'For direct analysis, call calc_METHOD with the method you want to use.')

    @staticmethod
    def calc_MAE(meas, sim):
        """
        Calculates the MAE (mean absolute error)
        for the given numpy array of measured and simulated data.
        :param meas: array
            Array with measurement data
        :param sim: array
            Array with simulation data
        :return: MAE: float
            MAE os the given data.
        """
        return skmetrics.mean_absolute_error(meas, sim)

    @staticmethod
    def calc_R2(meas, sim):
        """
        Calculates the MAE (mean absolute error)
        for the given numpy array of measured and simulated data.
        :param meas: array
            Array with measurement data
        :param sim: array
            Array with simulation data
        :return: MAE: float
            R2 of the given data.
        """
        return 1 - skmetrics.r2_score(meas, sim)

    @staticmethod
    def calc_MSE(meas, sim):
        """
        Calculates the MSE (mean square error)
        for the given numpy array of measured and simulated data.
        :param meas: array
            Array with measurement data
        :param sim: array
            Array with simulation data
        :return: MSE: float
            MSE of the given data.
        """
        return skmetrics.mean_squared_error(meas, sim)

    @staticmethod
    def calc_RMSE(meas, sim):
        """
        Calculates the RMSE (root mean square error)
        for the given numpy array of measured and simulated data.
        :param meas: array
            Array with measurement data
        :param sim: array
            Array with simulation data
        :return: RMSE: float
            RMSE of the given data.
        """
        return np.sqrt(skmetrics.mean_squared_error(meas, sim))

    @staticmethod
    def calc_NRMSE(meas, sim):
        """
        Calculates the NRMSE (normalized root mean square error)
        for the given numpy array of measured and simulated data.
        :param meas: array
            Array with measurement data
        :param sim: array
            Array with simulation data
        :return: NRMSE: float
            NRMSE of the given data.
        """
        # Check if NRMSE can be calculated
        if (np.max(meas) - np.min(meas)) == 0:
            raise ValueError("The given measurement data's maximum is equal to "
                             "it's minimum. This makes the calculation of the "
                             "NRMSE impossible. Choose another method.")
        else:
            return np.sqrt(skmetrics.mean_squared_error(meas, sim)) \
                   / (np.max(meas) - np.min(meas))

    @staticmethod
    def calc_CVRMSE(meas, sim):
        """
        Calculates the CVRMSE (variance of root mean square error)
        for the given numpy array of measured and simulated data.
        :param meas: array
            Array with measurement data
        :param sim: array
            Array with simulation data
        :return: CVRMSE: float
            CVRMSE of the given data.
        """
        # Check if CVRMSE can be calculated
        if np.mean(meas) == 0:
            raise ValueError("The given measurement data has a mean of 0. "
                             "This makes the calculation of the CVRMSE impossible. "
                             "Choose another method.")
        else:
            return np.sqrt(skmetrics.mean_squared_error(meas, sim)) / np.mean(meas)
