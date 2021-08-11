"""
Module containing data types to enable an automatic usage of
different other modules in the Python package.
"""
import warnings
import logging
from typing import Union
from copy import deepcopy
import pandas as pd
import numpy as np
from ebcpy import TimeSeriesData
from ebcpy.utils.statistics_analyzer import StatisticsAnalyzer
from ebcpy.preprocessing import convert_datetime_index_to_float_index
# pylint: disable=I1101
__version__ = "0.1.6"

logger = logging.getLogger(__name__)


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
    :param str statistical_measure:
        Measure to calculate the scalar of the objective,
        One of the supported methods in
        ebcpy.utils.statistics_analyzer.StatisticsAnalyzer
        e.g. RMSE, MAE, NRMSE
    :param list weightings:
        Values between 0 and 1 to account for multiple Goals to be evaluated.
        If multiple goals are selected, and weightings is None, each
        weighting will be equal to 1/(Number of goals).
        The weighting is scaled so that the sum will equal 1.
    """

    # Set default string for measurement reference
    meas_tag_str = "meas"
    sim_tag_str = "sim"

    def __init__(self,
                 meas_target_data: Union[TimeSeriesData, pd.DataFrame],
                 variable_names: dict,
                 statistical_measure: str,
                 weightings: list = None):
        """Initialize class-objects and check correct input."""

        # Open the meas target data:
        if not isinstance(meas_target_data, (TimeSeriesData, pd.DataFrame)):
            raise TypeError(f"Given meas_target_data is of type {type(meas_target_data).__name__} "
                            "but TimeSeriesData is required.")

        if not isinstance(variable_names, dict):
            raise TypeError(f"Given variable_names is of type {type(variable_names).__name__} "
                            f"but a dict is required.")

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
                raise TypeError(f"Variable {var_name} of variable_names has a value"
                                "neither being a dict, list or tuple.")
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
                                    f"Variable {meas_info} has mutliple tags, none of which "
                                    f"is specified as {self.meas_tag_str}.")
                if self.meas_tag_str in tags:
                    _columns.append((meas_info, self.meas_tag_str))
                else:
                    _columns.append((meas_info, tags[0]))
            elif isinstance(meas_info, tuple):
                _rename_cols_dict[meas_info[0]] = var_name
                _columns.append(meas_info)
            else:
                raise TypeError(f"Measurement Info on variable {var_name} is "
                                "neither of type string or tuple.")

        # Take the subset of the given tsd based on var_names and tags.
        self._tsd = meas_target_data[_columns].copy()

        # Rename all variables to the given var_name (key of self.variable_names)
        self._tsd = self._tsd.rename(columns=_rename_cols_dict, level=0)

        # Rename all tags to the default measurement name for consistency.
        tags = dict(zip(self._tsd.columns.levels[1],
                        [self.meas_tag_str for _ in range(len(_columns))]))
        self._tsd = self._tsd.rename(columns=tags, level=1)

        # Save the tsd to a tsd_ref object
        # Used to never lose the original dataframe.
        # _tsd may be altered by relevant intervals, this object never!
        self._tsd_ref = self._tsd.copy()

        # Set the statistical analyzer:
        self.statistical_measure = statistical_measure

        # Set the weightings, if not specified.
        self._num_goals = len(_columns)
        if weightings is None:
            self._weightings = np.array([1/self._num_goals
                                         for i in range(self._num_goals)])
        else:
            if not isinstance(weightings, (list, np.ndarray)):
                raise TypeError(f"weightings is of type {type(weightings).__name__} "
                                f"but should be of type list.")
            if len(weightings) != self._num_goals:
                raise IndexError(f"The given number of weightings ({len(weightings)}) "
                                 f"does not match the number of "
                                 f"goals ({self._num_goals})")
            self._weightings = np.array(weightings) / sum(weightings)

    def __str__(self):
        """Overwrite string method to present the Goals-Object more
        nicely."""
        return str(self._tsd)

    @property
    def statistical_measure(self) -> str:
        """The statistical measure of this Goal instance"""
        return self._stat_meas

    @statistical_measure.setter
    def statistical_measure(self, statistical_measure: str):
        self._stat_analyzer = StatisticsAnalyzer(method=statistical_measure)
        self._stat_meas = statistical_measure

    def eval_difference(self, verbose=False, penaltyfactor=1):
        """
        Evaluate the difference of the measurement and simulated data based on the
        chosen statistical_measure.

        :param boolean verbose:
            If True, a dict with difference-values of for all goals and the
            corresponding weightings is returned together with the total difference.
            This can be useful to better understand which goals is performing
            well in an optimization and which goals needs further is not performing well.
        :param float penaltyfactor:
            Muliplty result with this factor to account for
            penatlies of some sort.
        :return: float total_difference
            weighted ouput for all goals.
        """
        total_difference = 0
        _verbose_calculation = {}

        for i, goal_name in enumerate(self.variable_names.keys()):
            if self._tsd.isnull().values.any():
                raise ValueError("There are not valid values in the "
                                 "simulated target data. Probably the time "
                                 "interval of measured and simulated data "
                                 "are not equal. \nPlease check the frequencies "
                                 "in the toml file (output_interval & frequency).")
            _diff = self._stat_analyzer.calc(
                meas=self._tsd[(goal_name, self.meas_tag_str)],
                sim=self._tsd[(goal_name, self.sim_tag_str)]
            )
            # Apply penalty function
            _diff = _diff * penaltyfactor

            _verbose_calculation[self._weightings[i]] = _diff
            total_difference += self._weightings[i] * _diff

        if verbose:
            return total_difference, _verbose_calculation
        return total_difference

    def set_sim_target_data(self, sim_target_data):
        """Alter the object with new simulation data
        self._sim_target_data based on the given dataframe
        sim_target_data.

        :param TimeSeriesData sim_target_data:
            Object with simulation target data. This data should be
            the output of a simulation, hence "sim"-target-data.
        """
        # Start with the base
        self._tsd = self._tsd_ref.copy()
        # Check index type
        if not isinstance(sim_target_data.index, type(self._tsd.index)):
            raise IndexError(
                f"Given sim_target_data is using {type(sim_target_data.index).__name__}"
                f" as an index, but the reference results (measured-data) was declared"
                f" using the {type(self._tsd_ref.index).__name__}. Convert your"
                f" measured-data index to solve this error."
            )
        # Three critical cases may occur:
        # 1. sim_target_data is bigger (in len) than _tsd
        #   --> Only the first part is accepted
        # 2. sim_target_data is smaller than _tsd
        #   --> Missing values become NaN, which is fine. If no other function eliminates
        #       the NaNs, an error is raised when doing eval_difference().
        # 3. The index differs:
        #   --> All new values are NaN. However, this should raise an error, as an error
        #   in eval_difference would not lead back to this function.
        # Check if index matches in relevant intersection:
        sta = max(self._tsd.index[0], sim_target_data.index[0])
        sto = min(self._tsd.index[-1], sim_target_data.index[-1])
        if len(self._tsd.loc[sta:sto].index) != len(sim_target_data.loc[sta:sto].index):
            raise ValueError(f"Given indexes have different lengths "
                             f"({len(self._tsd.loc[sta:sto].index)} vs "
                             f"{len(sim_target_data.loc[sta:sto].index)}). "
                             f"Can't compare them. ")
        mask = self._tsd.loc[sta:sto].index != sim_target_data.loc[sta:sto].index
        if np.any(mask):
            diff = self._tsd.loc[sta:sto].index - sim_target_data.loc[sta:sto].index
            raise IndexError(f"Measured and simulated data differ on {np.count_nonzero(mask)}"
                             f" index points. Affected index part: {diff[mask]}. "
                             f"This will lead to errors in evaluation, "
                             f"hence we raise the error already here. "
                             f"Check output_interval, equidistant_output and "
                             f"frequency of measured data to find the reason for "
                             f"this error. The have to match.")

        # Resize simulation data to match to meas data
        for goal_name in self.variable_names.keys():
            _tsd_sim = sim_target_data.loc[sta:sto, self._sim_var_matcher[goal_name]]
            if len(_tsd_sim.columns) > 1:
                raise ValueError("Given sim_target_data contains multiple tags for variable "
                                 f"{self._sim_var_matcher[goal_name]}. "
                                 "Can't select one automatically.")
            self._tsd.loc[sta:sto, (goal_name, self.sim_tag_str)] = _tsd_sim.values
        # Sort the index for better visualisation
        self._tsd = self._tsd.sort_index(axis=1)

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
        _df_ref = self._tsd.copy()
        # Create initial False mask
        _mask = np.full(_df_ref.index.shape, False)
        # Dynamically make mask for multiple possible time-intervals
        for _start_time, _end_time in intervals:
            _mask = _mask | ((_df_ref.index >= _start_time) & (_df_ref.index <= _end_time))
        self._tsd = _df_ref.loc[_mask]

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

    def get_meas_frequency(self):
        """
        Get the frequency of the measurement data.

        :returns:
            float: Mean frequency of the index
        """
        mean, std = self._tsd.frequency
        if std >= 1e-8:
            logger.critical("The index of your measurement data is not "
                            f"equally sampled. The standard deviation is {mean.std()}."
                            "The may lead to errors when mapping measurements to simulation "
                            "results.")
        return mean


class TunerParas:
    """
    Class for tuner parameters.
    Tuner parameters are parameters of a model which are
    constant during simulation but are varied during calibration
    or other analysis.

    :param list names:
        List of names of the tuner parameters
    :param float,int initial_values:
        Initial values for optimization.
        Even though some optimization methods don't require an
        initial guess, specifying a initial guess based on
        expected values or experience is helpful to better
        check the results of the calibration
    :param list,tuple bounds:
        Tuple or list of float or ints for lower and upper bound to the tuner parameter.
        The bounds object is optional, however highly recommend
        for calibration or optimization in general. As soon as you
        tune parameters with different units, such as Capacity and
        heat conductivity, the solver will fail to find good solutions.


    Example:

    >>> tuner_paras = TunerParas(names=["C", "m_flow_2", "heatConv_a"],
    >>>                          initial_values=[5000, 0.02, 200],
    >>>                          bounds=[(4000, 6000), (0.01, 0.1), (10, 300)])
    >>> print(tuner_paras)
                initial_value      min     max    scale
    names
    C                 5000.00  4000.00  6000.0  2000.00
    m_flow_2             0.02     0.01     0.1     0.09
    heatConv_a         200.00    10.00   300.0   290.00
    """
    def __init__(self, names, initial_values, bounds=None):
        """Initialize class-objects and check correct input."""
        # Check if the given input-parameters are of correct format. If not, raise an error.
        for name in names:
            if not isinstance(name, str):
                raise TypeError(f"Given name is of type {type(name).__name__} "
                                "and not of type str.")
        # Check if all names are unique:
        if len(names) != len(set(names)):
            raise ValueError("Given names contain duplicates. "
                             "This will yield errors in later stages"
                             "such as calibration of sensitivity analysis.")
        try:
            # Calculate the sum, as this will fail if the elements are not float or int.
            sum(initial_values)
        except TypeError as err:
            raise TypeError("initial_values contains other "
                            "instances than float or int.") from err
        if len(names) != len(initial_values):
            raise ValueError(f"shape mismatch: names has length {len(names)}"
                             f" and initial_values {len(initial_values)}.")
        self._bounds = bounds
        if bounds is None:
            _bound_min = -np.inf
            _bound_max = np.inf
        else:
            if len(bounds) != len(names):
                raise ValueError(f"shape mismatch: bounds has length {len(bounds)} "
                                 f"and names {len(names)}.")
            _bound_min, _bound_max = [], []
            for bound in bounds:
                _bound_min.append(bound[0])
                _bound_max.append(bound[1])

        self._df = pd.DataFrame({"names": names,
                                 "initial_value": initial_values,
                                 "min": _bound_min,
                                 "max": _bound_max})
        self._df = self._df.set_index("names")
        self._set_scale()

    def __str__(self):
        """Overwrite string method to present the TunerParas-Object more
        nicely."""
        return str(self._df)

    def scale(self, descaled):
        """
        Scales the given value to the bounds of the tuner parameter between 0 and 1

        :param np.array,list descaled:
            Value to be scaled
        :return: np.array scaled:
            Scaled value between 0 and 1
        """
        # If no bounds are given, scaling is not possible--> descaled = scaled
        if self._bounds is None:
            return descaled
        _scaled = (descaled - self._df["min"])/self._df["scale"]
        if not all((_scaled >= 0) & (_scaled <= 1)):
            warnings.warn("Given descaled values are outside "
                          "of bounds. Automatically limiting "
                          "the values with respect to the bounds.")
        return np.clip(_scaled, a_min=0, a_max=1)

    def descale(self, scaled):
        """
        Converts the given scaled value to an descaled one.

        :param np.array,list scaled:
            Scaled input value between 0 and 1
        :return: np.array descaled:
            descaled value based on bounds.
        """
        # If no bounds are given, scaling is not possible--> descaled = scaled
        if not self._bounds:
            return scaled
        _scaled = np.array(scaled)
        if not all((_scaled >= 0-1e4) & (_scaled <= 1+1e4)):
            warnings.warn("Given scaled values are outside of bounds. "
                          "Automatically limiting the values with "
                          "respect to the bounds.")
        _scaled = np.clip(_scaled, a_min=0, a_max=1)
        return _scaled*self._df["scale"] + self._df["min"]

    @property
    def bounds(self):
        """Get property bounds"""
        return self._bounds

    def get_names(self):
        """Return the names of the tuner parameters"""
        return list(self._df.index)

    def get_initial_values(self):
        """Return the initial values of the tuner parameters"""
        return self._df["initial_value"].values

    def get_bounds(self):
        """Return the bound-values of the tuner parameters"""
        return self._df["min"].values, self._df["max"].values

    def get_value(self, name, col):
        """Function to get a value of a specific tuner parameter"""
        return self._df.loc[name, col]

    def set_value(self, name, col, value):
        """Function to set a value of a specific tuner parameter"""
        if not isinstance(value, (float, int)):
            raise ValueError(f"Given value is of type {type(value).__name__} "
                             "but float or int is required")
        if col not in ["max", "min", "initial_value"]:
            raise KeyError("Can only alter max, min and initial_value")
        self._df.loc[name, col] = value
        self._set_scale()

    def remove_names(self, names):
        """
        Remove gives list of names from the Tuner-parameters

        :param list names:
            List with names inside of the TunerParas-dataframe
        """
        self._df = self._df.drop(names)

    def _set_scale(self):
        self._df["scale"] = self._df["max"] - self._df["min"]
        if not self._df[self._df["scale"] <= 0].empty:
            raise ValueError(
                "The given lower bounds are greater equal "
                "than the upper bounds, resulting in a "
                f"negative scale: \n{str(self._df['scale'])}"
            )


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
        to be in an ascending order or be limited to
        the start_time and end_time parameters.
    :keyword (pd.DataFrame, ebcpy.data_types.TimeSeriesData) inputs:
        TimeSeriesData or DataFrame that holds
        input data for the simulation to run.
        The time-index should be float index and match the overall
        ranges set by start- and stop-time.
    :keyword dict input_kwargs:
        If inputs are provided, additional input keyword-args passed to the
        simulation API can be specified.
        Using FMUs, you don't need to specify anything.
        Using DymolaAPI, you have to specify 'table_name' and 'file_name'
    """

    def __init__(self, name, start_time, stop_time, goals=None,
                 tuner_paras=None, relevant_intervals=None, **kwargs):
        """Initialize class-objects and check correct input."""
        self.name = name
        self._start_time = start_time
        self.stop_time = stop_time
        self._goals = None
        self._tuner_paras = None
        if goals is not None:
            self.goals = goals
        if tuner_paras is not None:
            self.tuner_paras = tuner_paras
        if relevant_intervals is not None:
            self.relevant_intervals = relevant_intervals
        else:
            # Then all is relevant
            self.relevant_intervals = [(start_time, stop_time)]
        self._inputs = None
        inputs = kwargs.get('inputs', None)
        if inputs is not None:
            self.inputs = inputs  # Trigger the property setter
        self.input_kwargs = kwargs.get('input_kwargs', {})

    @property
    def name(self):
        """Get name of calibration class"""
        return self._name

    @name.setter
    def name(self, name: str):
        """Set name of calibration class"""
        if not isinstance(name, str):
            raise TypeError(f"Name of CalibrationClass is {type(name)} "
                            f"but has to be of type str")
        self._name = name

    @property
    def start_time(self) -> Union[float, int]:
        """Get start time of calibration class"""
        return self._start_time

    @start_time.setter
    def start_time(self, start_time: Union[float, int]):
        """Set start time of calibration class"""
        if not start_time <= self.stop_time:
            raise ValueError("The given start-time is "
                             "higher than the stop-time.")
        self._start_time = start_time

    @property
    def stop_time(self) -> Union[float, int]:
        """Get stop time of calibration class"""
        return self._stop_time

    @stop_time.setter
    def stop_time(self, stop_time: Union[float, int]):
        """Set stop time of calibration class"""
        if not self.start_time <= stop_time:
            raise ValueError("The given stop-time is "
                             "lower than the start-time.")
        self._stop_time = stop_time

    @property
    def tuner_paras(self) -> TunerParas:
        """Get the tuner parameters of the calibration-class"""
        return self._tuner_paras

    @tuner_paras.setter
    def tuner_paras(self, tuner_paras):
        """
        Set the tuner parameters for the calibration-class.

        :param tuner_paras: TunerParas
        """
        if not isinstance(tuner_paras, TunerParas):
            raise TypeError(f"Given tuner_paras is of type {type(tuner_paras).__name__} "
                            "but should be type TunerParas")
        self._tuner_paras = deepcopy(tuner_paras)

    @property
    def goals(self) -> Goals:
        """Get current goals instance"""
        return self._goals

    @goals.setter
    def goals(self, goals: Goals):
        """
        Set the goals object for the calibration-class.

        :param Goals goals:
            Goals-data-type
        """
        if not isinstance(goals, Goals):
            raise TypeError(f"Given goals parameter is of type {type(goals).__name__} "
                            "but should be type Goals")
        self._goals = deepcopy(goals)

    @property
    def relevant_intervals(self) -> list:
        """Get current relevant_intervals"""
        return self._relevant_intervals

    @relevant_intervals.setter
    def relevant_intervals(self, relevant_intervals: list):
        """Set current relevant_intervals"""
        self._relevant_intervals = relevant_intervals

    @property
    def inputs(self) -> Union[TimeSeriesData, pd.DataFrame]:
        """Get the inputs for this calibration class"""
        return self._inputs

    @inputs.setter
    def inputs(self, inputs: Union[TimeSeriesData, pd.DataFrame]):
        """Set the inputs for this calibration class"""
        # Check correct index:
        if not isinstance(inputs, (TimeSeriesData, pd.DataFrame)):
            raise TypeError(f"Inputs need to be either TimeSeriesData "
                            f"or pd.DataFrame, but you passed {type(inputs)}")
        if isinstance(inputs.index, pd.DatetimeIndex):
            inputs = convert_datetime_index_to_float_index(inputs)

        self._inputs = inputs


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
        # First create dictionary with all calibration classes
        if _name in temp_merged:
            temp_merged[_name]["intervals"] += cal_class.relevant_intervals
        else:
            temp_merged[_name] = {"goals": cal_class.goals,
                                  "tuner_paras": cal_class.tuner_paras,
                                  "intervals": cal_class.relevant_intervals,
                                  "inputs": cal_class.inputs
                                  }
    # Convert dict to actual calibration-classes
    cal_classes_merged = []
    for _name, values in temp_merged.items():
        # Flatten the list of tuples and get the start- and stop-values
        start_time = min(sum(values["intervals"], ()))
        stop_time = max(sum(values["intervals"], ()))
        cal_classes_merged.append(CalibrationClass(
            _name, start_time, stop_time,
            goals=values["goals"],
            tuner_paras=values["tuner_paras"],
            relevant_intervals=values["intervals"],
            inputs=values["inputs"])
        )

    return cal_classes_merged
