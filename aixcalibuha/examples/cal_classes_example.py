"""
Example file for the data_types module. The usage of classes inside
the data_types module should be clear when looking at the examples.
If not, please raise an issue.
"""
from ebcpy.examples import data_types_example
from aixcalibuha.examples import goals_example
from aixcalibuha import CalibrationClass, TunerParas


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
    # Load the tuner parameters and goals
    tuner_paras = data_types_example.setup_tuner_paras()
    goals = goals_example.setup_goals()
    # Set the tuner parameters and goals to all classes:
    for cal_class in calibration_classes:
        cal_class.set_tuner_paras(tuner_paras)
        cal_class.set_goals(goals)

    different_tuner_paras = TunerParas(names=["C", "heatConv_a"],
                                       initial_values=[5000, 200],
                                       bounds=[(4000, 6000), (10, 300)])
    calibration_classes[3].set_tuner_paras(different_tuner_paras)

    return calibration_classes
