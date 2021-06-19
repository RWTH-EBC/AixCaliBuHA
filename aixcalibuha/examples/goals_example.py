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
    >>> goals.set_relevant_time_intervals([(0, 100)])
    >>> print(round(goals.eval_difference(), 3))
    1.095
    """

    # Load example simTargetData and measTargetData:
    _filepath = os.path.dirname(__file__)
    sim_target_data = data_types.TimeSeriesData(os.path.join(_filepath,
                                                             "data",
                                                             "simTargetData.mat"))
    meas_target_data = data_types.TimeSeriesData(os.path.join(_filepath,
                                                              "data",
                                                              "ref_result.hdf"),
                                                 key="test")
    # Format: variable_names = {VARIABLE_NAME: [MEASUREMENT_NAME, SIMULATION_NAME]}
    variable_names = {"T_heater": ["measured_T_heater", "heater.heatPorts[1].T"],
                      "T_heater_1": ["measured_T_heater_1", "heater1.heatPorts[1].T"]}

    # Convert index to float to match the simulation output
    meas_target_data = preprocessing.convert_datetime_index_to_float_index(meas_target_data)
    # Setup the goals object
    goals = Goals(meas_target_data=meas_target_data,
                  variable_names=variable_names,
                  statistical_measure="RMSE",
                  weightings=[0.7, 0.3])
    goals.set_sim_target_data(sim_target_data)
    return goals


if __name__ == "__main__":
    GOALS = setup_goals()
    GOALS.set_relevant_time_intervals([(0, 100)])
    GOALS.eval_difference()
    print(GOALS)
