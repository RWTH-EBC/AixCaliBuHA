"""
Python package to calibrate models created in Modelica or possible
other simulation software.
"""

from ebcpy import data_types
# pylint: disable=I1101
__version__ = "0.1.3"


class CalibrationClass:
    """
    Class used for calibration of time-series data.

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
    :param list relevant_intervals:
        List with time-intervals relevant for the calibration.
        Each list element has to be a tuple with the first element being
        the start-time as float/int and the second item being the end-time
        of the interval as float/int.
        E.g:
        For a class with start_time=0 and stop_time=1000, given following intervals
        [(0, 100), [150, 200), (500, 600)]
        will only evaluate the data between 0-100, 150-200 and 500-600.
        The given intervals may overlap. Furthermore the intervals do not need
        to be in an ascending order or be limited to the start_time and end_time parameters.
    """

    goals = data_types.Goals
    tuner_paras = data_types.TunerParas
    relevant_intervals = []

    def __init__(self, name, start_time, stop_time, goals=None,
                 tuner_paras=None, relevant_intervals=None):
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
        if relevant_intervals:
            self.relevant_intervals = relevant_intervals
        else:
            # Then all is relevant
            self.relevant_intervals = [(start_time, stop_time)]

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
