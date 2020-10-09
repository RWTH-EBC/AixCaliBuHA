"""
Python package to calibrate models created in Modelica or possible
other simulation software.
"""

import pandas as pd
import numpy as np
from ebcpy import data_types
from ebcpy.utils import statistics_analyzer
import datetime
import warnings
warnings.simplefilter('always', UserWarning)    # for printing always warnings in loops
# pylint: disable=I1101
__version__ = "0.1.5"

class Preparation:      # Evtl. als Wrapper in eigenem Skript aufbauen, hier ist es Wrapper im selben Skript.
    """
    Prepare everything needed for perform a calibration.
        - Cleaning data
        - Setup the goals
        - Setup the tuner
        - Setup calibration classes

    :param: to add
    ...
    """

    def __init__(self, meas_target_data, sim_target_data, skip, SIM_API,
                 variable_names, weightings, number_cal_classes,
                 classes_toml, tuner_paras, timestamp):

        self.meas_target_data = meas_target_data
        self.SIM_API = SIM_API
        self.variable_names = variable_names
        self.weightings = weightings
        self.sim_target_data = sim_target_data
        self.number_cal_classes = number_cal_classes
        self.classes_toml = classes_toml
        self.tuner_paras = tuner_paras
        self.skip = skip
        self.timestamp = timestamp


    def setup_goals(self):
        """
        Setup of the Goals object.
        First, some simulated and measured target data is loaded.
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

        # Setup the goals object
        self.goals = Goals(meas_target_data=self.meas_target_data,
                           variable_names=self.variable_names,
                           weightings=self.weightings
                           )
        self.goals.set_sim_target_data(self.sim_target_data)

        # To match measured and simulated time interval (Removes all "NaN")
        self.goals.set_relevant_time_intervals([(self.SIM_API.sim_setup["startTime"], self.classes_toml[-1][2])])
        # Interpolate if the input data was resampled (see toml file) and extract goals data to dataframe
        # for check if compressor is on (see setup_calibration_classes)
        self.goals_data = self.goals.get_goals_data().interpolate()
        # self.classes_json[-1][2] is the stoptime of the LAST class element. This is important because we want
        # to evaluate the time interval to the end of this class and not to the stoptime specified in toml.

        return self.goals


    def setup_calibration_classes(self):
        """
        Setup of a list calibration classes.
        The measured data of the setup_goals example can
        be segmentized into two classes. You can either use
        classes from the segmentizer package or manually define
        classes of interest to you. In this example the we have
        a manual segmentation, as the example is fairly small.

        :param object tuner_paras:

        :param object goals:

        :return: List of calibration classes
        :rtype: list
        """
        # Define the basic time-intervals and names for the calibration-classes:
        # Read informations from toml file
        self.cal_classes = []
        # Create CalibrationClass object
        for i in range(self.number_cal_classes):
            if self.classes_toml[i][2] > self.SIM_API.sim_setup['stopTime']:
                self.classes_toml[i][2] = self.SIM_API.sim_setup['stopTime']
                warnings.warn('The stoptime of calibration class {} \nexceeds the stoptime adjusted during'
                              ' the extraction of the data and is changed to {} seconds.'
                              .format(self.classes_toml[i][0], self.SIM_API.sim_setup['stopTime']))
            self.cal_classes.append(CalibrationClass(name=self.classes_toml[i][0],
                                                     start_time=self.classes_toml[i][1],
                                                     stop_time=self.classes_toml[i][2]))

        # Set the tuner parameters and goals to all classes:
        for cal_class in self.cal_classes:
            cal_class.set_tuner_paras(self.tuner_paras)
            cal_class.set_goals(self.goals)
        # Use different tuner paras if necessary (example: in cool down phase (calibrationclass 4 in the example) there is no m_flow --> no tuning of m_flow possible)
        # different_tuner_paras = data_types.TunerParas(names=["C", "heatConv_a"],
        #                                               initial_values=[5000, 200],
        #                                               bounds=[(4000, 6000), (10, 300)])
        # calibration_classes[3].set_tuner_paras(different_tuner_paras)


        # Check if there are any values of the targets bigger then zero. For example the compressor could be
        # switched off in observed period. If so: no calibration for the current iteration step is done.

            ## %% TO-DO: What if compressor was shut off in just one calibration class? skip the class not whole cali.

            # Get goals of current calibration class
            goals_data_current_class = self.goals_data.loc[cal_class.start_time: cal_class.stop_time]

            # Iterate over all target variables of the class
            for i, target in enumerate(cal_class.goals.get_goals_list()):
                # get mean of goals to evaluate
                mean_of_meas_data = goals_data_current_class[target]["meas"].mean()
                if mean_of_meas_data == 0:
                    warnings.warn("Class '{}': The target '{}' seems \nto contain no data bigger then zero during the "
                                  "current considered period (affects day {}). This will cause problems for scale-"
                                  "independent metrics like CVRMSE,\n because the mean of the measured targes will be 0."
                                  " The calibration will be executed for the next period."
                                  .format(cal_class.name, target, self.timestamp.date()))
                    # Switch skipping on
                    # self.skip = True
                    self.skip['skip'] = True
                    self.skip['cal_class']
                    return self.cal_classes, self.skip

        return self.cal_classes, self.skip

# class Goals:
#     """
#     Class for one or multiple goals. Used to evaluate the
#     difference between current simulation and measured data
#
#     :param (ebcpy.data_types.TimeSeriesData, pd.DataFrame) meas_target_data:
#         The dataset of the measurement. It acts as a point of reference
#         for the simulation output. If the dimensions of the given DataFrame and later
#         added simulation-data are not equal, an error is raised.
#         Has to hold all variables listed under the MEASUREMENT_NAME variable in the
#         variable_names dict.
#     :param dict variable_names:
#         A dictionary to construct the goals-DataFrame using pandas MultiIndex-Functionality.
#         The dict has to follow the structure.
#         variable_names = {VARIABLE_NAME: [MEASUREMENT_NAME, SIMULATION_NAME]}
#             - VARIABLE_NAME: A string which holds the actual name
#                 of the variable you use as a goal.
#                 E.g.: VARIABLE_NAME="Temperature_Condenser_Outflow"
#             - MEASUREMENT_NAME: Is either a string or a tuple. Hold the name the variable
#                 has inside the given meas_target_data. If you want to specify a tag you have
#                 to pass a tuple, like: (MEASUREMENT_NAME, TAG_NAME). Else just pass a string.
#                 E.g.: MEASUREMENT_NAME="HydraulicBench[4].T_Out" or
#                       MEASUREMENT_NAME=("HydraulicBench[4].T_Out", "preprocessed")
#             - SIMULATION_NAME is either a string or a tuple, just like MEASUREMENT_NAME.
#                 E.g. (for Modelica): SIMULATION_NAME="HeatPump.Condenser.Vol.T"
#         You may use a tuple instead of a list OR a dict
#         with key "meas" for measurement and key "sim" for simulation. These options may be
#         relevant for your own code readability.
#         E.g. variable_names = {VARIABLE_NAME: {"meas":MEASUREMENT_NAME,
#                                                "sim": SIMULATION_NAME}}
#
#     :param list weightings:
#         Values between 0 and 1 to account for multiple Goals to be evaluated.
#         If multiple goals are selected, and weightings is None, each
#         weighting will be equal to 1/(Number of goals).
#         The weighting is scaled so that the sum will equal 1.
#     """
#
#     # Set default string for measurement reference
#     meas_tag_str = "meas"
#     sim_tag_str = "sim"
#
#     def __init__(self, meas_target_data, variable_names, weightings=None):
#         """Initialize class-objects and check correct input."""
#
#         # Open the meas target data:
#         if not isinstance(meas_target_data, (data_types.TimeSeriesData, pd.DataFrame)):
#             raise TypeError("Given meas_target_data is of type {} but TimeSeriesData "
#                             "is required.".format(type(meas_target_data).__name__))
#
#         if not isinstance(variable_names, dict):
#             raise TypeError("Given variable_names is of type {} but a dict is "
#                             "required.".format(type(variable_names).__name__))
#
#         # Extract the measurement-information out of the dict.
#         self.variable_names = variable_names
#
#         # Used to speed up the frequently used set_sim_target_data function
#         self._sim_var_matcher = {}
#         _columns = []  # Used to extract relevant part of df
#
#         _rename_cols_dict = {}
#         for var_name, meas_sim_info in self.variable_names.items():
#             # First extract the information about the measurement out of the dict
#             if isinstance(meas_sim_info, dict):
#                 meas_info = meas_sim_info[self.meas_tag_str]
#                 self._sim_var_matcher[var_name] = meas_sim_info[self.sim_tag_str]
#             elif isinstance(meas_sim_info, (list, tuple)):
#                 meas_info = meas_sim_info[0]
#                 self._sim_var_matcher[var_name] = meas_sim_info[1]
#             else:
#                 raise TypeError("Variable {} of variable_names has a value"
#                                 "neither being a dict, list or tuple.".format(var_name))
#             # Now get the info to extract the values out of the given tsd
#             # Convert string with into a list of tuples containing the relevant tag.
#             # If mulitple tags exist, and the default tag (self.meas_tag_str)
#             # is not present, an error is raised.
#             if isinstance(meas_info, str):
#                 if isinstance(meas_target_data[meas_info], pd.Series):
#                     raise TypeError("Given meas_target_data contains columns without a tag."
#                                     "Please only pass MultiIndex-DataFrame objects.")
#                 tags = meas_target_data[meas_info].columns
#                 _rename_cols_dict[meas_info] = var_name
#                 if len(tags) != 1 and self.meas_tag_str not in tags:
#                     raise TypeError("Not able to automatically select variables and tags. "
#                                     "Variable {} has mutliple tags, none of which "
#                                     "is specified as {}.".format(meas_info, self.meas_tag_str))
#                 elif self.meas_tag_str in tags:
#                     _columns.append((meas_info, self.meas_tag_str))
#                 else:
#                     _columns.append((meas_info, tags[0]))
#             elif isinstance(meas_info, tuple):
#                 _rename_cols_dict[meas_info[0]] = var_name
#                 _columns.append(meas_info)
#             else:
#                 raise TypeError("Measurement Info on variable {} is "
#                                 "neither of type string or tuple.".format(var_name))
#
#         # Take the subset of the given tsd based on var_names and tags.
#         self._tsd = meas_target_data[_columns].copy()
#
#         # Rename all variables to the given var_name (key of self.variable_names)
#         self._tsd = self._tsd.rename(columns=_rename_cols_dict, level=0)
#
#         # Rename all tags to the default measurement name for consistency.
#         d = dict(zip(self._tsd.columns.levels[1],
#                      [self.meas_tag_str for _ in range(len(_columns))]))
#         self._tsd = self._tsd.rename(columns=d, level=1)
#
#         # Save the tsd to a tsd_ref object
#         # Used to never lose the original dataframe.
#         # _tsd may be altered by relevant intervals, this object never!
#         self._tsd_ref = self._tsd.copy()
#
#         # Set the weightings, if not specified.
#         self._num_goals = len(_columns)
#         if weightings is None:
#             self._weightings = np.array([1/self._num_goals for i in range(self._num_goals)])
#         else:
#             if not isinstance(weightings, (list, np.ndarray)):
#                 raise TypeError("weightings is of type {} but should be of type"
#                                 " list.".format(type(weightings).__name__))
#             if len(weightings) != self._num_goals:
#                 raise IndexError("The given number of weightings ({}) does not match the number"
#                                  " of goals ({})".format(len(weightings), self._num_goals))
#             self._weightings = np.array(weightings) / sum(weightings)
#
#     def __str__(self):
#         """Overwrite string method to present the Goals-Object more
#         nicely."""
#         return str(self._tsd)
#
#     def eval_difference(self, statistical_measure, verbose=False):
#         """
#         Evaluate the difference of the measurement and simulated data based on the
#         given statistical_measure.
#
#         :param str statistical_measure:
#             Method supported by ebcpy.utils.statistics_analyzer.StatisticsAnalyzer, e.g. RMSE
#         :param boolean verbose:
#             If True, a dict with difference-values of for all goals and the
#             corresponding weightings is returned together with the total difference.
#             This can be useful to better understand which goals is performing
#             well in an optimization and which goals needs further is not performing well.
#         :return: float total_difference
#             weighted ouput for all goals.
#         """
#         stat_analyzer = statistics_analyzer.StatisticsAnalyzer(statistical_measure)
#         total_difference = 0
#         _verbose_calculation = {}
#
#         for i, goal_name in enumerate(self.variable_names.keys()):
#             _diff = stat_analyzer.calc(meas=self._tsd[(goal_name, self.meas_tag_str)],
#                                        sim=self._tsd[(goal_name, self.sim_tag_str)])
#             _verbose_calculation[self._weightings[i]] = _diff
#             total_difference += self._weightings[i] * _diff
#
#         if verbose:
#             return total_difference, _verbose_calculation
#         else:
#             return total_difference
#
#     def set_sim_target_data(self, sim_target_data):
#         """Alter the object with new simulation data
#         self._sim_target_data based on the given dataframe
#         sim_target_data.
#
#         :param TimeSeriesData sim_target_data:
#             Object with simulation target data. This data should be
#             the output of a simulation, hence "sim"-target-data.
#         """
#         # Check correct input to avoid wrong results at all cost
#         if not isinstance(sim_target_data, (data_types.TimeSeriesData, pd.DataFrame)):
#             raise TypeError("Given meas_target_data is of type {} but TimeSeriesData "
#                             "is required.".format(type(sim_target_data).__name__))
#
#         if not isinstance(sim_target_data.index, type(self._tsd_ref.index)):
#             raise IndexError("Given sim_target_data is using {} as an index, but the "
#                              "reference results (measured-data) was declared using the "
#                              "{}. Convert your measured-data index to solve this error. "
#                              "".format(type(sim_target_data.index).__name__,
#                                        type(self._tsd_ref.index).__name__))
#
#         for goal_name in self.variable_names.keys():
#             # Three critical cases may occur:
#             # 1. sim_target_data is bigger (in len) than _tsd_ref
#             #   --> Only the first part is accepted
#             # 2. sim_target_data is smaller than _tsd_ref
#             #   --> Missing values become NaN, which is fine. If no other function eliminates
#             #       the NaNs, an error is raised when doing eval_difference().
#             # 3. The index differs:
#             #   --> All new values are NaN. However, this should raise an error, as an error
#             #   in eval_difference would not lead back to this function.
#             self._tsd_ref[(goal_name, self.sim_tag_str)] = \
#                 sim_target_data[self._sim_var_matcher[goal_name]]
#         # Sort the index for better visualisation
#         self._tsd_ref = self._tsd_ref.sort_index(axis=1)
#         self._tsd = self._tsd_ref.copy()
#
#     def set_relevant_time_intervals(self, intervals):
#         """
#         For many calibration-uses cases, different time-intervals of the measured
#         and simulated data are relevant. Set the interval to be used with this function.
#         This will change both measured and simulated data. Therefore, the eval_difference
#         function can be called at every moment.
#
#         :param list intervals:
#             List with time-intervals. Each list element has to be a tuple
#             with the first element being the start_time as float or int and
#             the second item being the end_time of the interval as float or int.
#             E.g:
#             [(0, 100), [150, 200), (500, 600)]
#         """
#         _df_ref = self._tsd_ref.copy()
#         # Create initial False mask
#         _mask = np.full(_df_ref.index.shape, False)
#         # Dynamically make mask for multiple possible time-intervals
#         for _start_time, _end_time in intervals:
#             _mask = _mask | ((_df_ref.index >= _start_time) & (_df_ref.index <= _end_time))
#         # TODO: Is the data temporarly deleted if a segment is applied? Maybe we need a base-tsd and a current-tsd like _curr_tsd
#         self._tsd = _df_ref.loc[_mask]
#
#     def get_goals_list(self):
#         """Get the internal list containing all goals."""
#         return list(self.variable_names.keys())
#
#     def get_goals_data(self):
#         """Get the current time-series-data object."""
#         return self._tsd.copy()
#
#     def save(self):
#         config = {"meas_taget_data": "??",
#                   "variable_names": self.variable_names,
#                   "weightings": self._weightings}
#         return config
#

class Goals:
    """
    Class for one or multiple goals. Used to evaluate the
    difference between current simulation and measured data

    :param (ebcpy.data_types.TimeSeriesData, pd.DataFrame) meas_target_data:
        The dataset of the measurement. It acts as a point of reference
        for the simulation output. If the dimensions of the given DataFrame and later
        added simulation-data are not equal, an error is raised.
        Has to hold all variables listed under the MEASUREMENT_NAME variable in the
        variable_names dict.
    :param dict variable_names:
        A dictionary to construct the goals-DataFrame using pandas MultiIndex-Functionality.
        The dict has to follow the structure.
        variable_names = {VARIABLE_NAME: [MEASUREMENT_NAME, SIMULATION_NAME]}
            - VARIABLE_NAME: A string which holds the actual name
                of the variable you use as a goal.
                E.g.: VARIABLE_NAME="Temperature_Condenser_Outflow"
            - MEASUREMENT_NAME: Is either a string or a tuple. Hold the name the variable
                has inside the given meas_target_data. If you want to specify a tag you have
                to pass a tuple, like: (MEASUREMENT_NAME, TAG_NAME). Else just pass a string.
                E.g.: MEASUREMENT_NAME="HydraulicBench[4].T_Out" or
                      MEASUREMENT_NAME=("HydraulicBench[4].T_Out", "preprocessed")
            - SIMULATION_NAME is either a string or a tuple, just like MEASUREMENT_NAME.
                E.g. (for Modelica): SIMULATION_NAME="HeatPump.Condenser.Vol.T"
        You may use a tuple instead of a list OR a dict
        with key "meas" for measurement and key "sim" for simulation. These options may be
        relevant for your own code readability.
        E.g. variable_names = {VARIABLE_NAME: {"meas":MEASUREMENT_NAME,
                                               "sim": SIMULATION_NAME}}

    :param list weightings:
        Values between 0 and 1 to account for multiple Goals to be evaluated.
        If multiple goals are selected, and weightings is None, each
        weighting will be equal to 1/(Number of goals).
        The weighting is scaled so that the sum will equal 1.
    """

    # Set default string for measurement reference
    meas_tag_str = "meas"
    sim_tag_str = "sim"

    def __init__(self, meas_target_data, variable_names, weightings=None):
        """Initialize class-objects and check correct input."""

        # Open the meas target data:
        if not isinstance(meas_target_data, (data_types.TimeSeriesData, pd.DataFrame)):
            raise TypeError("Given meas_target_data is of type {} but TimeSeriesData "
                            "is required.".format(type(meas_target_data).__name__))

        if not isinstance(variable_names, dict):
            raise TypeError("Given variable_names is of type {} but a dict is "
                            "required.".format(type(variable_names).__name__))

        # Extract the measurement-information out of the dict.
        self.variable_names = variable_names

        # Used to speed up the frequently used set_sim_target_data function
        self._sim_var_matcher = {}
        _columns = []  # Used to extract relevant part of df

        _rename_cols_dict = {}
        for var_name, meas_sim_info in self.variable_names.items():
            # First extract the information about the measurement out of the dict
            if isinstance(meas_sim_info, dict):
                meas_info = meas_sim_info[self.meas_tag_str]
                self._sim_var_matcher[var_name] = meas_sim_info[self.sim_tag_str]
            elif isinstance(meas_sim_info, (list, tuple)):
                meas_info = meas_sim_info[0]
                self._sim_var_matcher[var_name] = meas_sim_info[1]
            else:
                raise TypeError("Variable {} of variable_names has a value"
                                "neither being a dict, list or tuple.".format(var_name))
            # Now get the info to extract the values out of the given tsd
            # Convert string with into a list of tuples containing the relevant tag.
            # If mulitple tags exist, and the default tag (self.meas_tag_str)
            # is not present, an error is raised.
            if isinstance(meas_info, str):
                if isinstance(meas_target_data[meas_info], pd.Series):
                    raise TypeError("Given meas_target_data contains columns without a tag."
                                    "Please only pass MultiIndex-DataFrame objects.")
                tags = meas_target_data[meas_info].columns
                _rename_cols_dict[meas_info] = var_name
                if len(tags) != 1 and self.meas_tag_str not in tags:
                    raise TypeError("Not able to automatically select variables and tags. "
                                    "Variable {} has mutliple tags, none of which "
                                    "is specified as {}.".format(meas_info, self.meas_tag_str))
                elif self.meas_tag_str in tags:
                    _columns.append((meas_info, self.meas_tag_str))
                else:
                    _columns.append((meas_info, tags[0]))
            elif isinstance(meas_info, tuple):
                _rename_cols_dict[meas_info[0]] = var_name
                _columns.append(meas_info)
            else:
                raise TypeError("Measurement Info on variable {} is "
                                "neither of type string or tuple.".format(var_name))

        # Take the subset of the given tsd based on var_names and tags.
        self._tsd = meas_target_data[_columns].copy()

        # Rename all variables to the given var_name (key of self.variable_names)
        self._tsd = self._tsd.rename(columns=_rename_cols_dict, level=0)

        # Rename all tags to the default measurement name for consistency.
        d = dict(zip(self._tsd.columns.levels[1],
                     [self.meas_tag_str for _ in range(len(_columns))]))
        self._tsd = self._tsd.rename(columns=d, level=1)

        # Save the tsd to a tsd_ref object
        # Used to never lose the original dataframe.
        # _tsd may be altered by relevant intervals, this object never!
        self._tsd_ref = self._tsd.copy()

        # Set the weightings, if not specified.
        self._num_goals = len(_columns)
        if weightings is None:
            self._weightings = np.array([1/self._num_goals for i in range(self._num_goals)])
        else:
            if not isinstance(weightings, (list, np.ndarray)):
                raise TypeError("weightings is of type {} but should be of type"
                                " list.".format(type(weightings).__name__))
            if len(weightings) != self._num_goals:
                raise IndexError("The given number of weightings ({}) does not match the number"
                                 " of goals ({})".format(len(weightings), self._num_goals))
            self._weightings = np.array(weightings) / sum(weightings)

    def __str__(self):
        """Overwrite string method to present the Goals-Object more
        nicely."""
        return str(self._tsd)

    def eval_difference(self, statistical_measure, verbose=False, penaltyfactor=1):
        """
        Evaluate the difference of the measurement and simulated data based on the
        given statistical_measure.

        :param str statistical_measure:
            Method supported by statistics_analyzer.StatisticsAnalyzer, e.g. RMSE
        :param boolean verbose:
            If True, a dict with difference-values of for all goals and the
            corresponding weightings is returned together with the total difference.
            This can be useful to better understand which goals is performing
            well in an optimization and which goals needs further is not performing well.
        :param float penaltyfactor:
            ...to add..
        :return: float total_difference
            weighted ouput for all goals.
        """
        stat_analyzer = statistics_analyzer.StatisticsAnalyzer(statistical_measure)
        total_difference = 0
        _verbose_calculation = {}

        for i, goal_name in enumerate(self.variable_names.keys()):
            if self._tsd.isnull().values.any():
                raise ValueError("There are not valid values in the simulated target data. Probably the time interval"
                                 " of measured and simulated data are not equal. \nPlease check the frequencies"
                                 " in the toml file (outputInterval & frequency).")
            _diff = stat_analyzer.calc(meas=self._tsd[(goal_name, self.meas_tag_str)],
                                       sim=self._tsd[(goal_name, self.sim_tag_str)])
            # Apply penalty function
            _diff = _diff * penaltyfactor

            _verbose_calculation[self._weightings[i]] = _diff
            total_difference += self._weightings[i] * _diff

        if verbose:
            return total_difference, _verbose_calculation
        else:
            return total_difference

    def set_sim_target_data(self, sim_target_data):
        """Alter the object with new simulation data
        self._sim_target_data based on the given dataframe
        sim_target_data.

        :param TimeSeriesData sim_target_data:
            Object with simulation target data. This data should be
            the output of a simulation, hence "sim"-target-data.
        """
        if not isinstance(sim_target_data.index, type(self._tsd_ref.index)):
            raise IndexError("Given sim_target_data is using {} as an index, but the "
                             "reference results (measured-data) was declared using the "
                             "{}. Convert your measured-data index to solve this error. "
                             "".format(type(sim_target_data.index).__name__,
                                       type(self._tsd_ref.index).__name__))

        for goal_name in self.variable_names.keys():
            # Three critical cases may occur:
            # 1. sim_target_data is bigger (in len) than _tsd_ref
            #   --> Only the first part is accepted
            # 2. sim_target_data is smaller than _tsd_ref
            #   --> Missing values become NaN, which is fine. If no other function eliminates
            #       the NaNs, an error is raised when doing eval_difference().
            # 3. The index differs:
            #   --> All new values are NaN. However, this should raise an error, as an error
            #   in eval_difference would not lead back to this function.
            self._tsd_ref[(goal_name, self.sim_tag_str)] = \
                sim_target_data[self._sim_var_matcher[goal_name]]
        # Sort the index for better visualisation
        self._tsd_ref = self._tsd_ref.sort_index(axis=1)
        self._tsd = self._tsd_ref.copy()

    def set_relevant_time_intervals(self, intervals):
        """
        For many calibration-uses cases, different time-intervals of the measured
        and simulated data are relevant. Set the interval to be used with this function.
        This will change both measured and simulated data. Therefore, the eval_difference
        function can be called at every moment.

        :param list intervals:
            List with time-intervals. Each list element has to be a tuple
            with the first element being the start_time as float or int and
            the second item being the end_time of the interval as float or int.
            E.g:
            [(0, 100), [150, 200), (500, 600)]
        """
        _df_ref = self._tsd_ref.copy()
        # Interpolate if resampling is activated (see toml file for more informations)
        _df_ref = _df_ref.interpolate()
        # Create initial False mask
        _mask = np.full(_df_ref.index.shape, False)
        # Dynamically make mask for multiple possible time-intervals
        for _start_time, _end_time in intervals:
            _mask = _mask | ((_df_ref.index >= _start_time) & (_df_ref.index <= _end_time))
        # TODO: Is the data temporarly deleted if a segment is applied? Maybe we need a base-tsd and a current-tsd like _curr_tsd
        self._tsd = _df_ref.loc[_mask]
        # # Interpolate if resampling is activated (see toml file for more informations)
        # self._tsd = self._tsd.interpolate()


    def get_goals_list(self):
        """Get the internal list containing all goals."""
        return list(self.variable_names.keys())

    def get_goals_data(self):
        """Get the current time-series-data object."""
        return self._tsd.copy()

    def get_sim_var_names(self):
        """Get the names of the simulation variables.

        :returns list sim_var_names:
            Names of the simulation variables as a list
        """
        return list(self._sim_var_matcher.values())


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

    goals = None
    tuner_paras = None
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
        if not isinstance(goals, Goals):
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

    def use_specific_tuners(self, selection):
        """
        Use specific tuner parameter, if user knows which tuner parameter have an impact and which have not.
        If just one "TunerParas" class object is defined (see data_types.TunerParas()) the iteration for every
        calibration class is unnecessary because the first iteration causes an overwriting process of the
        TunerParas object in the current CalibrationClass object with "set_tuner_paras".

        :param list selection:
            List of strings with desired tunerparameters for calibration
        """

        removing_names = []
        removing_bounds = []
        for i, name in enumerate(self.tuner_paras.get_names()):
            if not name in selection:
                removing_bounds.append(self.tuner_paras.bounds[i])
                removing_names.append(name)
        self.tuner_paras.remove_names(removing_names)
        self.set_tuner_paras(self.tuner_paras)

    # def use_specific_tuners(self, selection):
    #     """
    #     Use specific tuner parameter, if user knows which tuner parameter have an impact and which have not.
    #
    #     :param list selection:
    #         List of strings with desired tunerparameters for calibration
    #     """
    #
    #     for num_class, cal_class in enumerate(self.cal_classes):
    #         removing_names = []
    #         removing_bounds = []
    #         for i, name in enumerate(self.tuner_paras.get_names()):
    #             if not name in selection:
    #                 removing_bounds.append(self.tuner_paras.bounds[i])
    #                 removing_names.append(name)
    #         self.tuner_paras.remove_names(removing_names)
    #         cal_class.set_tuner_paras(self.tuner_paras)


def merge_calibration_classes(calibration_classes):
    """
    Given a list of multiple calibration-classes, this function merges given
    objects by the "name" attribute. Relevant intervals are set, in order
    to maintain the start and stop-time info.
    :param list calibration_classes:
        List containing multiple CalibrationClass-Objects
    :return: list cal_classes_merged:
        A list containing one CalibrationClass-Object for each different
        "name" of class.

    Example:
    >>> cal_classes = [CalibrationClass("on", 0, 100),
    >>>                CalibrationClass("off", 100, 200),
    >>>                CalibrationClass("on", 200, 300)]
    >>> merged_classes = merge_calibration_classes(cal_classes)
    Is equal to:
    >>> merged_classes = [CalibrationClass("on", 0, 300,
    >>>                                    relevant_intervals=[(0,100), (200,300)]),
    >>>                   CalibrationClass("off", 100, 200)]
    """
    # Use a dict for easy name-access
    temp_merged = {}
    for cal_class in calibration_classes:
        _name = cal_class.name
        if _name in temp_merged:
            temp_merged[_name]["intervals"] += cal_class.relevant_intervals
        else:
            temp_merged[_name] = {"goals": cal_class.goals,
                                  "tuner_paras": cal_class.tuner_paras,
                                  "intervals": cal_class.relevant_intervals
                                  }
    # Convert dict to actual calibration-classes
    cal_classes_merged = []
    for _name, values in temp_merged.items():
        # Flatten the list of tuples and get the start- and stop-values
        start_time = min(sum(values["intervals"], ()))
        stop_time = max(sum(values["intervals"], ()))
        cal_classes_merged.append(CalibrationClass(_name, start_time, stop_time,
                                                   goals=values["goals"],
                                                   tuner_paras=values["tuner_paras"],
                                                   relevant_intervals=values["intervals"]))

    return cal_classes_merged
