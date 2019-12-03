"""
Module with classes and function to help visualize
different processes inside the framework. Both plots
and print-function/log-function will be implemented here.
The Visualizer Class inherits the Logger class, as logging
will always be used as a default.
"""
import os
from ebcpy import data_types
from ebcpy.utils.visualizer import Logger
import matplotlib.pyplot as plt
import numpy as np


class CalibrationLogger(Logger):
    """Base class for showing the process of functions in
        this Framework with print-statements and saving everything
        relevant as a log-file.

        :param str,os.path.normpath cd:
            Directory where to store the output of the Logger and possible
            child-classes. If the given directory can not be created, an error
            will be raised.
        :param str name:
            Name of the reason of logging, e.g. classification, processing etc.
        :param aixcal.data_types.TunerParas tuner_paras:
            TunerParas class used in the calibration-process.
    """

    # Instantiate dummy parameters
    tuner_paras = None
    integer_prec = 4  # Number of integer parts
    decimal_prec = 6
    _counter_calibration = 0  # Number of function calls of calibration
    _prec = decimal_prec
    _width = integer_prec + decimal_prec + 1  # Calculate the actual width

    def __init__(self, cd, name, tuner_paras):
        """Instantiate class parameters"""
        super().__init__(cd, name)
        self.set_tuner_paras(tuner_paras)

    def _set_prec_and_with_for_tuner_paras(self):
        if self.tuner_paras.bounds is None:
            self.integer_prec = 4  # Number of integer parts
        else:
            bounds_min, bounds_max = self.tuner_paras.get_bounds()
            maximal_value = max(max(bounds_max), max(abs(bounds_min)))
            self.integer_prec = len(str(int(maximal_value)))
        self._counter_calibration = 0  # Number of function calls of calibration
        self._width = self.integer_prec + self.decimal_prec + 1  # Calculate the actual width

    def calibration_callback_func(self, xk, obj):
        """
        Logs the current values of the objective function.

        :param np.array xk:
            Array with the current values of the calibration
        :param float obj:
            Current objective value.
        """
        xk_descaled = self.tuner_paras.descale(xk)
        self._counter_calibration += 1
        info_string = self._get_tuner_para_values_as_string(xk_descaled, obj)
        self.log(info_string)

    def save_calibration_result(self, res, model_name, statistical_measure, **kwargs):
        """
        Process the result, re-run the simulation and generate
        a logFile for the minimal quality measurement

        :param scipy.optimize.minimize.result res:
            Result object of the minimization
        :param str model_name:
            Name of the model being calibrated
        :param str statistical_measure:
            Statistical measure used for calibration.
        """
        result_log = "Results for calibration of model: {}\n".format(model_name)
        result_log += "Used statistical measure {}\n".format(statistical_measure)
        result_log += "Final parameter values:\n"
        result_log += "{}\n".format(self._get_tuner_para_names_as_string(statistical_measure))
        final_values = self._get_tuner_para_values_as_string(self.tuner_paras.descale(res.x),
                                                             res.fun)
        result_log += "{}\n".format(final_values)
        result_log += "Number of iterations: {}\n".format(self._counter_calibration)
        result_log += "Result of optimization-framework:\n{}".format(res)
        self.log(result_log)
        self._counter_calibration = 0

    def set_tuner_paras(self, tuner_paras):
        """
        Set the currently used TunerParas object to use the information for logging.

        :param tuner_paras: aixcal.data_types.TunerParas
        """
        if not isinstance(tuner_paras, data_types.TunerParas):
            raise TypeError("Given tuner_paras is of type {} but type"
                            "TunerParas is needed.".format(type(tuner_paras).__name__))
        self.tuner_paras = tuner_paras
        self._set_prec_and_with_for_tuner_paras()

    def log_initial_names(self, statistical_measure):
        """Function to log the initial names and the statistical measure
        before calibration."""
        _text_initial_names = self._get_tuner_para_names_as_string(statistical_measure)
        self.log(_text_initial_names)

    def _get_tuner_para_names_as_string(self, statistical_measure):
        """
        Returns a string with the names of current tunerParameters

        :return: str info_string
            The desired string
        """
        initial_names = list(self.tuner_paras.get_names())

        info_string = "{0:9s}".format("Iteration")

        for ini_name in initial_names:
            # Limit string length to a certain amount.
            # The full name has to be displayed somewhere else
            if len(ini_name) > self._width:
                num_dots = len(ini_name) - self._width
                if num_dots > 3:
                    num_dots = 3
                formatted_name = "."*num_dots + ini_name[-(self._width-num_dots):]
            else:
                formatted_name = ini_name
            info_string += "   {0:{width}s}".format(formatted_name, width=self._width)
        # Add string for qualitative measurement used (e.g. NRMSE, MEA etc.)
        info_string += "   {0:{width}s}".format(statistical_measure, width=self._width)
        return info_string

    def _get_tuner_para_values_as_string(self, xk_descaled, obj):
        """
        Returns a string with the values of current tuner parameters
        as well as the objective value.

        :param np.array xk_descaled:
            Array with the current values of the calibration, descaled to bounds
        :param float obj:
            Current objective value.
        :return: str
        The desired string
        """
        # This will limit the number of iterations to 999999999 (for correct format).
        # More iterations will most likely never be used.
        info_string = '{0:9d}'.format(self._counter_calibration)

        for x_value in xk_descaled:
            info_string += "   {0:{width}.{prec}f}".format(x_value,
                                                           width=self._width,
                                                           prec=self._prec)
        # Add the last return value of the objective function.
        info_string += "   {0:{width}.{prec}f}".format(obj, width=self._width,
                                                       prec=self._prec)
        return info_string


class CalibrationVisualizer(CalibrationLogger):
    """More advanced class to not only log ongoing function
    evaluations but also show the process of the functions
    by plotting interesting causalities and saving these plots.

    :param aixcal.data_types.Goals goals:
        Goals object with the data for plotting
    """

    # Setup dummy parameters so class-functions now the type of those later created objects:
    _n_cols_goals, _n_rows_goals, _n_cols_tuner, _n_rows_tuner = 1, 1, 1, 1
    plt.ioff()  # Turn of interactive mode. Only
    fig_tuner, ax_tuner = None, None
    fig_goal, ax_goal = None, None
    fig_obj, ax_obj = None, None
    goals = None
    _num_goals = 0
    show_plot = True

    def __init__(self, cd, name, tuner_paras, **kwargs):
        """Instantiate class parameters"""

        # Instantiate the logger:
        super().__init__(cd, name, tuner_paras)
        if isinstance(kwargs.get("show_plot"), bool):
            self.show_plot = kwargs.get("show_plot")
        # Setup the figures:
        self.calibrate_new_class(name, tuner_paras, kwargs.get("goals"))

    def calibrate_new_class(self, name, tuner_paras, goals=None):
        """Function to setup the figures for a new class of calibration.
        This function is called when instantiating this Class. If you
        uses continuuos calibration classes, call this function before
        starting the next calibration-class.

        :param str name:
            Name of the reason of logging, e.g. classification, processing etc.
        :param aixcal.data_types.TunerParas tuner_paras:
            TunerParas class used in the calibration-process.
        :param aixcal.data_types.Goals goals:
            Goals object with the data for plotting
        """

        self.set_tuner_paras(tuner_paras)

        if goals is not None:
            self.set_goals(goals)

        plt.close("all")

        # %% Set-up figure for objective-plotting
        self.fig_obj, self.ax_obj = plt.subplots(1, 1)
        self.fig_obj.suptitle(name + ": Objective")
        self.ax_obj.set_ylabel("objective")
        self.ax_obj.set_xlabel("Number iterations")
        # If the changes are small, it seems like the plot does
        # not fit the printed values. This boolean assures that no offset is used.
        self.ax_obj.ticklabel_format(useOffset=False)

        # %% Setup Tuner-Paras figure
        # Make a almost quadratic layout based on the number of tuner-parameters evolved.
        num_tuners = len(self.tuner_paras.get_names())
        self._n_cols_tuner = int(np.floor(np.sqrt(num_tuners)))
        self._n_rows_tuner = int(np.ceil(num_tuners / self._n_cols_tuner))
        self.fig_tuner, self.ax_tuner = plt.subplots(self._n_rows_tuner, self._n_cols_tuner,
                                                     squeeze=False, sharex=True)
        self.fig_tuner.suptitle(name + ": Tuner Parameters")
        self._plot_tuner_parameters(for_setup=True)

        # %% Setup Goals figure
        # Only a dummy, as the figure is recreated at every iteration
        if self.goals is not None:
            self._num_goals = len(self.goals.get_goals_list())
            self._n_cols_goals = int(np.floor(np.sqrt(self._num_goals)))
            self._n_rows_goals = int(np.ceil(self._num_goals / self._n_cols_goals))
            self.fig_goal, self.ax_goal = plt.subplots(self._n_rows_goals, self._n_cols_goals,
                                                       squeeze=False, sharex=True)
            self.fig_goal.suptitle(name + ": Goals")

    def calibration_callback_func(self, xk, obj):
        """
        Logs the current values of the objective function.

        :param np.array xk:
            Array with the current values of the calibration
        :param float obj:
            Current objective value.
        :param aixcal.data_types.Goals goals:
            Goals object with the data for plotting
        """
        # Call the logger function to print and log
        super().calibration_callback_func(xk, obj)
        # Plot the current objective value
        self.ax_obj.plot(self._counter_calibration, obj, "ro")

        # Plot the tuner parameters
        self._plot_tuner_parameters(xk=xk)

        # Plot the measured and simulated data
        if self.goals is not None:
            self._plot_goals()

        if self.show_plot:
            plt.draw()
            plt.pause(1e-5)

    def save_calibration_result(self, res, model_name, statistical_measure, **kwargs):
        """
        Process the result, re-run the simulation and generate
        a logFile for the minimal quality measurement

        :param scipy.optimize.minimize.result res:
            Result object of the minimization
        :param str model_name:
            Name of the model being calibrated
        :param str statistical_measure:
            Statistical measure used for calibration.
        :keyword str file_type:
            svg, pdf or png
        """
        file_type = "svg"
        if isinstance(kwargs.get("file_type"), str):
            file_type = kwargs.get("file_type")
        super().save_calibration_result(res, model_name, statistical_measure)

        filepath_tuner = os.path.join(self.cd, "tuner_parameter_plot.%s" % file_type)
        filepath_obj = os.path.join(self.cd, "objective_plot.%s" % file_type)
        self.fig_tuner.savefig(filepath_tuner)
        self.fig_obj.savefig(filepath_obj)
        plt.close("all")

    def set_goals(self, goals):
        """
        Set the currently used Goals object to use the information for logging.

        :param aixcal.data_types.Goals goals:
            Goals to be set to the object
        """
        if not isinstance(goals, data_types.Goals):
            raise TypeError("Given goals is of type {} but type"
                            "Goals is needed.".format(type(goals).__name__))
        self.goals = goals

    def _plot_tuner_parameters(self, xk=None, for_setup=False):
        """
        Plot the tuner parameter values history for better user interaction

        :param np.array xk:
            current iterate, scaled.
        :param bool for_setup:
            True if the function is called to initialize the calibration
        """
        tuner_counter = 0
        for row in range(self._n_rows_tuner):
            for col in range(self._n_cols_tuner):
                cur_ax = self.ax_tuner[row][col]
                if tuner_counter >= len(self.tuner_paras.get_names()):
                    cur_ax.axis("off")
                else:
                    tuner_para_name = self.tuner_paras.get_names()[tuner_counter]
                    if for_setup:
                        cur_ax.set_ylabel(tuner_para_name)
                        max_value = self.tuner_paras.get_value(tuner_para_name, "max")
                        min_value = self.tuner_paras.get_value(tuner_para_name, "min")
                        ini_val = self.tuner_paras.get_value(tuner_para_name, "initial_value")
                        cur_ax.axhline(max_value, color="r")
                        cur_ax.axhline(min_value, color="r")
                        cur_ax.plot(self._counter_calibration, ini_val, "bo")
                    if xk is not None:
                        cur_val = self.tuner_paras.descale(xk)[tuner_counter]
                        cur_ax.plot(self._counter_calibration, cur_val, "bo")
                    tuner_counter += 1

    def _plot_goals(self):
        """Plot the measured and simulated data for the current iterate"""

        _goals = self.goals.get_goals_list()
        goal_counter = 0
        for row in range(self._n_rows_goals):
            for col in range(self._n_cols_goals):
                cur_ax = self.ax_goal[row][col]
                cur_ax.cla()
                if goal_counter >= self._num_goals:
                    cur_ax.axis("off")
                else:
                    cur_goal = _goals[goal_counter]
                    goal_names = self.goals.get_goal_names(goal_counter)
                    cur_ax.plot(cur_goal.sim,
                                label=goal_names["sim_name"] + "_sim", linestyle="--", color="r")
                    cur_ax.plot(cur_goal.meas,
                                label=goal_names["meas_name"] + "_meas", color="b")
                    cur_ax.legend(loc="upper right")
                    cur_ax.set_xlabel("Time / s")
                goal_counter += 1