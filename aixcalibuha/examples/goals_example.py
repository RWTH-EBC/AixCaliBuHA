"""
Example file for the Goals class. The usage of the class
should be clear when looking at the examples.
If not, please raise an issue.
"""
import os

from ebcpy import data_types, preprocessing
from aixcalibuha import Goals


def setup_goals():
    """
    Example setup of the Goals object.
    First, some simulated and measured target data is loaded from the
    example data.
    Then the goals object is instantiated. Please refer to the
    Goals documentation on the meaning of the parameters.


    :return: Goals object
    :rtype: aixcalibuha.Goals

    Example:

    >>> goals = setup_goals()
    >>> dif = goals.eval_difference(statistical_measure="RMSE")
    >>> print(round(dif, 3))
    1.055
    """

    # Load example simTargetData and measTargetData:
    _filepath = os.path.dirname(__file__)
    sim_target_data = data_types.TimeSeriesData(os.path.join(_filepath,
                                                             "data",
                                                             "simTargetData.mat"))
    meas_target_data = data_types.TimeSeriesData(os.path.join(_filepath,
                                                              "data",
                                                              "ref_result.hdf"))
    variable_names = {"T_heater": ["measured_T_heater", "heater.heatPorts[1].T"],
                      "T_heater_1": ["measured_T_heater_1", "heater1.heatPorts[1].T"]}

    # Convert index to float to match the simulation output
    meas_target_data = preprocessing.convert_datetime_index_to_float_index(meas_target_data)
    # Setup the goals object
    goals = Goals(meas_target_data=meas_target_data,
                  variable_names=variable_names,
                  weightings=[0.7, 0.3])
    goals.set_sim_target_data(sim_target_data)
    return goals


if __name__=="__main__":
    goals = setup_goals()
    print(goals)
