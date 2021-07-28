"""
Example file for the data_types module. The usage of classes inside
the data_types module should be clear when looking at the examples.
If not, please raise an issue.
"""
import pathlib
from ebcpy import data_types, preprocessing, TimeSeriesData
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
    # As the examples should work, and the cal_class example uses the other examples,
    # we will test it here:
    data_dir = pathlib.Path(__file__).parent.joinpath("data")
    sim_target_data = TimeSeriesData(data_dir.joinpath("PumpAndValveSimulation.mat"))
    meas_target_data = TimeSeriesData(data_dir.joinpath("PumpAndValve.hdf"), key="example")

    # Setup three variables for different format of setup
    variable_names = {"T": ["TCapacity", "heatCapacitor.T"],
                      "m_flow": {"meas": "m_flow_valve", "sim": "valve.flowPort_a.m_flow"}}

    # Convert index to float to match the simulation output
    meas_target_data.to_float_index()
    # Setup the goals object
    goals = Goals(meas_target_data=meas_target_data,
                  variable_names=variable_names,
                  statistical_measure="NRMSE",
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
    # Specify the tuner parameters and goals
    tuner_paras = TunerParas(names=["speedRamp.duration", "valveRamp.duration"],
                             initial_values=[0.1, 0.1],
                             bounds=[(0.1, 10), (0.1, 10)])

    different_tuner_paras = TunerParas(names=["speedRamp.duration"],
                                       initial_values=[0.1],
                                       bounds=[(0.1, 10)])

    goals = setup_goals()

    # Define the basic time-intervals and names for the calibration-classes:
    calibration_classes = [
        CalibrationClass(name="Heat up", start_time=0, stop_time=1,
                         goals=goals, tuner_paras=tuner_paras),
        CalibrationClass(name="cool down", start_time=1, stop_time=2,
                         goals=goals, tuner_paras=tuner_paras),
        CalibrationClass(name="stationary", start_time=2, stop_time=10,
                         goals=goals, tuner_paras=different_tuner_paras)
    ]

    return calibration_classes


if __name__ == "__main__":
    CAL_CLASSES = setup_calibration_classes()
    print(CAL_CLASSES)
