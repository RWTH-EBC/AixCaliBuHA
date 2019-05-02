import numpy as np
import sklearn.metrics as skmetrics


class StatisticsAnalyzer():
    def __init__(self):
        raise NotImplementedError


def calc_stat_values(meas, sim):
    """
    Calculates multiple statistical values for the given numpy array of measured and simulated data.
    Calculates:
    MAE(Mean absolute error), RMSE(root mean square error), R2(coefficient of determination), CVRMSE(variance of RMSE), NRMSE(Normalized RMSE)
    :param meas:
    Array with measurement data
    :param sim:
    Array with simulation data
    :return: stat_values: dict
    Containing all calculated statistical values
    """
    statistical_measures = {"MAE": skmetrics.mean_absolute_error(meas, sim),
                   "RMSE": np.sqrt(skmetrics.mean_squared_error(meas, sim)),
                   "R2": 1 - skmetrics.r2_score(meas, sim)}
    # Check if CVRMSE can be calculated
    if np.mean(meas) != 0:
        statistical_measures["CVRMSE"] = statistical_measures["RMSE"] / np.mean(meas)
    # Check if NRMSE can be calculated
    if (np.max(meas) - np.min(meas)) != 0:
        statistical_measures["NRMSE"] = statistical_measures["RMSE"] / (np.max(meas) - np.min(meas))
    return statistical_measures
