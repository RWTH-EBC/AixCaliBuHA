"""
Python package to calibrate models created in Modelica or possible
other simulation software.
"""

from ebcpy import data_types
# pylint: disable=I1101
__version__ = "0.1.2"


class CalibrationClass:
    """
    Class used for continuous calibration.

    :param str name:
        Name of the class, e.g. 'device on'
    :param float,int start_time:
        Time at which the class starts
    :param float,int stop_time:
        Time at which the class ends
    :param Goals goals:
        Goals parameters which are relevant in this class.
        As this class may be used in the classifier, a Goals-Class
        may not be available at all times and can be added later.
    :param TunerParas tuner_paras:
        As this class may be used in the classifier, a TunerParas-Class
        may not be available at all times and can be added later.
    """

    goals = data_types.Goals
    tuner_paras = data_types.TunerParas

    def __init__(self, name, start_time, stop_time, goals=None, tuner_paras=None):
        """Initialize class-objects and check correct input."""
        if not start_time <= stop_time:
            raise ValueError("The given start-time is higher than the stop-time.")
        if not isinstance(name, str):
            raise TypeError("Name of CalibrationClass is {} but"
                            " has to be of type str".format(type(name)))
        self.name = name
        self.start_time = start_time
        self.stop_time = stop_time
        if goals:
            self.set_goals(goals)
        if tuner_paras:
            self.set_tuner_paras(tuner_paras)

    def set_goals(self, goals):
        """
        Set the goals object for the calibration-class.

        :param Goals goals:
            Goals-data-type
        """
        if not isinstance(goals, data_types.Goals):
            raise TypeError("Given goals parameter is of type {} but should be "
                            "type Goals".format(type(goals).__name__))
        self.goals = goals

    def set_tuner_paras(self, tuner_paras):
        """
        Set the tuner parameters for the calibration-class.

        :param TunerParas tuner_paras:
            TunerParas to be set to calibration class
        """
        if not isinstance(tuner_paras, data_types.TunerParas):
            raise TypeError("Given tuner_paras is of type {} but should be "
                            "type TunerParas".format(type(tuner_paras).__name__))
        self.tuner_paras = tuner_paras
