"""
Module with classes and function to help visualize
different processes inside the framework. Both plots
and print-function/log-function will be implemented here.
The Visualizer Class inherits the Logger class, as logging
will always be used as a default.
"""
import os
from datetime import datetime
from aixcal import data_types
import matplotlib.pyplot as plt
import numpy as np
import pydot
from IPython.display import Image
from sklearn.externals.six import StringIO
import sklearn.tree as sktree
import seaborn


class Logger:
    """Base class for showing the process of functions in
    this Framework with print-statements and saving everything
    relevant as a log-file.
    :param cd: str, os.path.normpath
        Directory where to store the output of the Logger and possible
        child-classes. If the given directory can not be created, an error
        will be raised.
    :param name: str
        Name of the reason of logging, e.g. classification, processing etc.
    """
    def __init__(self, cd, name):
        """Instantiate class parameters"""

        self.cd = cd
        if not os.path.isdir(self.cd):
            os.makedirs(self.cd)
        # Setup the logger
        self.filepath_log = os.path.join(cd, "%s.log" % name)
        self.name = name

        # Check if previous logs exist and create some spacer
        _spacer = "-" * 150
        if os.path.isfile(self.filepath_log):
            with open(self.filepath_log, "a+") as log_file:
                log_file.seek(0)
                if log_file.read() != "":
                    log_file.write("\n" + _spacer)

        self.integer_prec = 4  # Number of integer parts
        self.decimal_prec = 6
        self._counter_calibration = 0  # Number of function calls of calibration
        self._prec = self.decimal_prec
        self._width = self.integer_prec + self.decimal_prec + 1  # Calculate the actual width

    def log(self, text):
        """
        Logs the given text to the given log.
        :param text:
        :return:
        """
        print(text)
        datestring = datetime.now().strftime("%d.%m.%Y-%H:%M:%S")
        with open(self.filepath_log, "a+") as log_file:
            log_file.write("\n{}: {}".format(datestring, text))

    def show_log(self):
        """Function to open the log-file.
        May be used at the end of a process.
        """
        os.system(self.filepath_log)

class Visualizer(Logger):
    """More advanced class to not only log ongoing function
    evaluations but also show the process of the functions
    by plotting interesting causalities and saving these plots."""

    def __init__(self, cd, name):
        """Instantiate class parameters"""
        super().__init__(cd, name)
        plt.ioff()  # Turn of interactive mode. Only


class CalibrationLogger(Logger):
    """Base class for showing the process of functions in
        this Framework with print-statements and saving everything
        relevant as a log-file.
        :param cd: str, os.path.normpath
            Directory where to store the output of the Logger and possible
            child-classes. If the given directory can not be created, an error
            will be raised.
        :param name: str
            Name of the reason of logging, e.g. classification, processing etc.
        :param tuner_paras: aixcal.data_types.TunerParas
            TunerParas class used in the calibration-process.
    """

    # Instantiate dummy parameters
    tuner_paras = None

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
        self.decimal_prec = 6
        self._counter_calibration = 0  # Number of function calls of calibration
        self._prec = self.decimal_prec
        self._width = self.integer_prec + self.decimal_prec + 1  # Calculate the actual width

    def calibration_callback_func(self, xk, obj):
        """
        Logs the current values of the objective function.
        :param xk: np.array
            Array with the current values of the calibration
        :param obj: float
            Current objective value.
        """
        xk_descaled = self.tuner_paras.descale(xk)
        self._counter_calibration += 1
        info_string = self._get_tuner_para_values_as_string(xk_descaled, obj)
        self.log(info_string)

    def save_calibration_result(self, res, model_name, statistical_measure):
        """
        Process the result, re-run the simulation and generate
        a logFile for the minimal quality measurement
        :param res: scipy.optimize.minimize.result
            Result object of the minimization
        :param model_name: str
            Name of the model being calibrated
        :param statistical_measure: str
            Statistical measure used for calibration.
        """
        result_log = "Results for calibration of model: {}\n".format(model_name)
        result_log += "Used statistical measure {}\n".format(statistical_measure)
        result_log += "Final parameter values:\n"
        result_log += "{}\n".format(self._get_tuner_para_names_as_string(statistical_measure))
        final_values = self._get_tuner_para_values_as_string(self.tuner_paras.descale(res.x), res.fun)
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
        :return: str
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
        :param xk_descaled: np.array
            Array with the current values of the calibration, descaled to bounds
        :param obj: float
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
    :param goals: aixcal.data_types.Goals
        Goals object with the data for plotting
    """

    # Setup dummy parameters so class-functions now the type of those later created objects:
    _n_cols_goals, _n_rows_goals, _n_cols_tuner, _n_rows_tuner = 1, 1, 1, 1
    plt.ioff()  # Turn of interactive mode. Only
    fig_tuner, ax_tuner = plt.subplots(_n_rows_tuner, _n_cols_tuner, squeeze=False)
    fig_goal, ax_goal = plt.subplots(_n_rows_goals, _n_cols_goals, squeeze=False)
    fig_obj, ax_obj = plt.subplots(1, 1, squeeze=False)
    goals = None
    _num_goals = 0

    def __init__(self, cd, name, tuner_paras, goals=None, show_plot=True):
        """Instantiate class parameters"""

        # Instantiate the logger:
        super().__init__(cd, name, tuner_paras)
        self.show_plot = show_plot
        # Setup the figures:
        self.calibrate_new_class(name, tuner_paras, goals)

    def calibrate_new_class(self, name, tuner_paras, goals=None):
        """Function to setup the figures for a new class of calibration.
        This function is called when instantiating this Class. If you
        uses continuuos calibration classes, call this function before
        starting the next calibration-class.
        :param name: str
            Name of the reason of logging, e.g. classification, processing etc.
        :param tuner_paras: aixcal.data_types.TunerParas
            TunerParas class used in the calibration-process.
        :param goals: aixcal.data_types.Goals
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
        :param xk: np.array
            Array with the current values of the calibration
        :param obj: float
            Current objective value.
        :param goals: aixcal.data_types.Goals
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

    def save_calibration_result(self, res, model_name, statistical_measure, file_type="svg"):
        """
        Process the result, re-run the simulation and generate
        a logFile for the minimal quality measurement
        :param res: scipy.optimize.minimize.result
            Result object of the minimization
        :param model_name: str
            Name of the model being calibrated
        :param statistical_measure: str
            Statistical measure used for calibration.
        :param file_type:
            svg, pdf or png
        :return:
        """
        super().save_calibration_result(res, model_name, statistical_measure)
        filepath_tuner = os.path.join(self.cd, "tuner_parameter_plot.%s" % file_type)
        filepath_obj = os.path.join(self.cd, "objective_plot.%s" % file_type)
        self.fig_tuner.savefig(filepath_tuner)
        self.fig_obj.savefig(filepath_obj)
        plt.close("all")

    def set_goals(self, goals):
        """
        Set the currently used Goals object to use the information for logging.
        :param goals: aixcal.data_types.Goals
        """
        if not isinstance(goals, data_types.Goals):
            raise TypeError("Given goals is of type {} but type"
                            "Goals is needed.".format(type(goals).__name__))
        self.goals = goals

    def _plot_tuner_parameters(self, xk=None, for_setup=False):
        """
        Plot the tuner parameter values history for better user interaction
        :param xk: np.array
            current iterate, scaled.
        :param for_setup: bool
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


class ClassifierVisualizer(Visualizer):
    """Visualizer class used for all classification processes.
    More advanced class to not only log ongoing function
    evaluations but also show the process of the functions
    by plotting interesting causalities and saving these plots."""

    def __init__(self, cd, name):
        """Instantiate instance attributes"""
        super().__init__(cd, name)

    def export_decision_tree_image(self, dtree, variable_list):
        """
        Saves the given dtree object by exporting it
        via graphviz to a png image
        :param dtree: DecisionTree
        :param variable_list: list
            List with names of decision-variables
        :return:
        """
        # Save the created tree as a png.
        try:
            # Visualization decision tree
            dot_data = StringIO()
            sktree.export_graphviz(dtree,
                                   out_file=dot_data,
                                   feature_names=variable_list, filled=True, rounded=True)
            graph = pydot.graph_from_dot_data(dot_data.getvalue())
            Image(graph[0].create_png())  # Creating the image needs some time
            plt.show(graph[0])
            graph[0].write_png(self.cd+'/tree_plot.png')
        except OSError:
            self.log("ERROR: Can not export the decision tree, "
                     "please install graphviz on your machine.")

    def plot_decision_tree(self, df, class_list):
        """Visualization pair plot (df is data frame with whole X values (train and test)
        This function takes a long time to be executed.
        :param df: pd.DataFrame
        :param class_list: list
            List with names for classes.
        """

        seaborn.pairplot(df, hue=class_list)
        plt.savefig(self.cd + '/pairplot.png', bbox_inches='tight', dpi=400)
