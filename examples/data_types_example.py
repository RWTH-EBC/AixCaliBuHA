"""
Example file for the data_types module. The usage of classes inside
the data_types module should be clear when looking at the examples.
If not, please raise an issue.
"""
import os
from ebcpy import data_types, preprocessing
from aixcalibuha import CalibrationClass, TunerParas, Goals


def setup_goals():
    """
    Example setup of the Goals object.
    First, some simulated and measured target data is loaded from the
    example data.
    Then the goals object is instantiated. Please refer to the
    Goals documentation on the meaning of the parameters.


    :return: Goals object
    :rtype: aixcalibuha.Goals
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


def setup_calibration_classes():
    """
    Example setup of a list calibration classes.
    The measured data of the setup_goals example can
    be segmentized into two classes. You can either use
    classes from the segmentizer package or manually define
    classes of interest to you. In this example the we have
    a manual segmentation, as the example is fairly small.

    :return: List of calibration classes
    :rtype: list
    """
    # Define the basic time-intervals and names for the calibration-classes:
    calibration_classes = [
        CalibrationClass(name="Heat up", start_time=0, stop_time=90),
        CalibrationClass(name="Heat up", start_time=110, stop_time=200),
        CalibrationClass(name="stationary", start_time=200, stop_time=400),
        CalibrationClass(name="cool down", start_time=400, stop_time=500),
        CalibrationClass(name="stationary", start_time=500, stop_time=600),
    ]
    # Specify the tuner parameters and goals
    tuner_paras = TunerParas(names=["C", "m_flow_2", "heatConv_a"],
                             initial_values=[5000, 0.02, 200],
                             bounds=[(4000, 6000), (0.01, 0.1), (10, 300)])
    goals = setup_goals()
    # Set the tuner parameters and goals to all classes:
    for cal_class in calibration_classes:
        cal_class.tuner_paras = tuner_paras
        cal_class.goals = goals

    different_tuner_paras = TunerParas(names=["C", "heatConv_a"],
                                       initial_values=[5000, 200],
                                       bounds=[(4000, 6000), (10, 300)])
    calibration_classes[3].tuner_paras = different_tuner_paras

    return calibration_classes


if __name__ == "__main__":
    CAL_CLASSES = setup_calibration_classes()
    print(CAL_CLASSES)
