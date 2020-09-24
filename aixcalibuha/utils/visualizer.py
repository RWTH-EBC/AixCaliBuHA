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
import aixcalibuha
from shutil import copyfile


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
        :param aixcalibuha.CalibrationClass calibration_class:
            Calibration class used in the calibration-process.
        :param str statistical_measure:
            Measurement used to evaluate the objective
    """

    # Instantiate dummy parameters
    calibration_class = aixcalibuha.CalibrationClass
    tuner_paras = data_types.TunerParas
    goals = aixcalibuha.Goals
    integer_prec = 4  # Number of integer parts
    decimal_prec = 6
    _counter_calibration = 0  # Number of function calls of calibration
    _prec = decimal_prec
    _width = integer_prec + decimal_prec + 1  # Calculate the actual width

    def __init__(self, cd, name, calibration_class, statistical_measure):
        """Instantiate class parameters"""
        super().__init__(cd, name)
        self.calibration_class = calibration_class
        self.set_tuner_paras(calibration_class.tuner_paras)
        self.statistical_measure = statistical_measure

    def _set_prec_and_with_for_tuner_paras(self):
        if self.tuner_paras.bounds is None:
            self.integer_prec = 4  # Number of integer parts
        else:
            bounds_min, bounds_max = self.tuner_paras.get_bounds()
            maximal_value = max(max(bounds_max), max(abs(bounds_min)))
            self.integer_prec = len(str(int(maximal_value)))
        self._counter_calibration = 0  # Number of function calls of calibration
        self._width = self.integer_prec + self.decimal_prec + 1  # Calculate the actual width

    def calibration_callback_func(self, xk, obj, verbose_information, penalty=None):
        """
        Logs the current values of the objective function.

        :param np.array xk:
            Array with the current values of the calibration
        :param float obj:
            Current objective value.
        :param dict verbose_information:
            A dict with difference-values of for all goals and the
            corresponding weightings
        :param float penalty:
            Penaltyfactor from current evaluation
        """
        xk_descaled = self.tuner_paras.descale(xk)
        self._counter_calibration += 1
        if penalty == None:
            info_string = self._get_tuner_para_values_as_string(xk_descaled, obj, verbose_information)
        else:
            info_string = self._get_tuner_para_values_as_string(xk_descaled, obj, verbose_information, penalty)
        self.log(info_string)

    def save_calibration_result(self, best_iterate, model_name, duration, itercount, **kwargs):
        """
        Process the result, re-run the simulation and generate
        a logFile for the minimal quality measurement

        :param dict best_iterate:
            Result object of the minimization
        :param str model_name:
            Name of the model being calibrated
        """
        result_log = "\nResults for calibration of model: {}\n".format(model_name)
        result_log += "Number of iterations: {}\n".format(self._counter_calibration)
        result_log += "Final parameter values:\n"
        # Set the iteration counter to the actual number of the best iteration is printed
        self._counter_calibration = best_iterate["Iterate"]
        result_log += "{}\n".format(self._get_tuner_para_names_as_string())
        final_values = self._get_tuner_para_values_as_string(best_iterate["Parameters"],
                                                             best_iterate["Objective"],
                                                             best_iterate["Unweighted Objective"],
                                                             best_iterate["Penaltyfactor"])
        result_log += "{}\n".format(final_values)
        self.log(result_log)
        self._counter_calibration = 0

    def calibrate_new_class(self, calibration_class, cd=None):
        """Function to setup the figures for a new class of calibration.
        This function is called when instantiating this Class. If you
        uses continuuos calibration classes, call this function before
        starting the next calibration-class.

        :param aixcalibuha.CalibrationClass calibration_class:
            Class holding information on names, tuner_paras, goals
            and time-intervals of calibration.
        :param str,os.path.normpath cd:
            Optional change in working directory to store files
        """
        if cd and os.path.isdir(cd):
            self.cd = cd
        if not os.path.exists(self.cd):
            os.makedirs(self.cd)

        self.calibration_class = calibration_class
        self.set_tuner_paras(calibration_class.tuner_paras)

        if calibration_class.goals is not None:
            self.set_goals(calibration_class.goals)

    def set_tuner_paras(self, tuner_paras):
        """
        Set the currently used TunerParas object to use the information for logging.

        :param tuner_paras: ebcpy.data_types.TunerParas
        """
        if not isinstance(tuner_paras, data_types.TunerParas):
            raise TypeError("Given tuner_paras is of type {} but type"
                            "TunerParas is needed.".format(type(tuner_paras).__name__))
        self.tuner_paras = tuner_paras
        self._set_prec_and_with_for_tuner_paras()

    def set_goals(self, goals):
        """
        Set the currently used Goals object to use the information for logging.

        :param ebcpy.aixcalibuha.Goals goals:
            Goals to be set to the object
        """
        if not isinstance(goals, aixcalibuha.Goals):
            raise TypeError("Given goals is of type {} but type"
                            "Goals is needed.".format(type(goals).__name__))
        self.goals = goals

    def log_initial_names(self):
        """Function to log the initial names and the statistical measure
        before calibration."""
        _text_initial_names = self._get_tuner_para_names_as_string()
        self.log(_text_initial_names)

    def log_intersection_of_tuners(self, intersected_tuner_parameters, itercount):
        """
        If an intersection for multiple classes occurs, an information about
        the statistics of the dataset has to be provided.

        :param dict intersected_tuner_parameters:
            Dict with cols being the name of the tuner parameter and the
            value being the list with all the different "best" values for
            the tuner parameter.
        """
        self.log("Multiple 'best' values for the following tuner parameters "
                 "were identified in different "
                 "classes:\n{}".format("\n".join(["{}: {}".format(tuner, values)
                                                  for tuner, values in intersected_tuner_parameters.items()])))

    def _get_tuner_para_names_as_string(self):
        """
        Returns a string with the names of current tunerParameters

        :return: str info_string
            The desired string
        """
        initial_names = list(self.tuner_paras.get_names())

        info_string = "{0:9s}".format("Iteration")

        # Names of tuner parameter
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
        info_string += "     {0:{width}s}".format(self.statistical_measure, width=self._width)
        info_string += "{}".format("penaltyfactor")
        info_string += "   {}".format("Unweighted {}".format(self.statistical_measure))
        return info_string

    def _get_tuner_para_values_as_string(self, xk_descaled, obj, unweighted_objective, penalty=None):
        """
        Returns a string with the values of current tuner parameters
        as well as the objective value.

        :param np.array xk_descaled:
            Array with the current values of the calibration, descaled to bounds
        :param float obj:
            Current objective value.
        :param dict unweighted_objective:
            Further information about the objective value of each individual goal
        :param None/float penalty:
            Penaltyfactor.
        :return: str
            The desired string.
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
        if penalty:
            info_string += "   {0:{width}.{prec}f}".format(penalty, width=self._width,
                                                           prec=self._prec-3)
        else:
            info_string += "        {}".format("-")
        _verbose_info = "= " + " + ".join(["{0:.{prec}}*{1:.{prec}}".format(weight, val, prec=4)
                                           for weight, val in unweighted_objective.items()])
        info_string += "          {}".format(_verbose_info)

        return info_string


class CalibrationVisualizer(CalibrationLogger):
    """More advanced class to not only log ongoing function
    evaluations but also show the process of the functions
    by plotting interesting causalities and saving these plots.

    :keyword boolean show_plot:
        If False, all created plots are not shown during calibration but only
        stored at the end of the process.
    :keyword boolean create_tsd_plot:
        If False, the plot of the time series data (goals) is not created and
        thus shown in during calibration. It therefore is also not stored, even if
        you set the save_tsd_plot keyword-argument to true.
    :keyword boolean save_tsd_plot:
        If True, at each iteration the created plot of the
        time-series is saved. This may make the process much slower

    """

    # Setup dummy parameters so class-functions now the type of those later created objects:
    _n_cols_goals, _n_rows_goals, _n_cols_tuner, _n_rows_tuner = 1, 1, 1, 1
    plt.ioff()  # Turn of interactive mode. Only
    fig_tuner, ax_tuner = None, None
    fig_goal, ax_goal = None, None
    fig_obj, ax_obj = None, None
    _num_goals = 0
    save_tsd_plot = False
    create_tsd_plot = True
    show_plot = True

    def __init__(self, cd, name, calibration_class, statistical_measure, **kwargs):
        """Instantiate class parameters"""

        # Instantiate the logger:
        super().__init__(cd, name, calibration_class, statistical_measure)
        # Set supported kwargs:
        if isinstance(kwargs.get("save_tsd_plot"), bool):
            self.save_tsd_plot = kwargs.get("save_tsd_plot")
        if isinstance(kwargs.get("create_tsd_plot"), bool):
            self.create_tsd_plot = kwargs.get("create_tsd_plot")
        if isinstance(kwargs.get("show_plot"), bool):
            self.show_plot = kwargs.get("show_plot")

    def calibrate_new_class(self, calibration_class, cd=None):
        """Function to setup the figures for a new class of calibration.
        This function is called when instantiating this Class. If you
        uses continuuos calibration classes, call this function before
        starting the next calibration-class.

        :param aixcalibuha.CalibrationClass calibration_class:
            Class holding information on names, tuner_paras, goals
            and time-intervals of calibration.
        :param str,os.path.normpath cd:
            Optional change in working directory to store files
        """
        super().calibrate_new_class(calibration_class, cd)

        name = calibration_class.name

        # Close all old figures to create new ones.
        plt.close("all")

        # %% Set-up figure for objective-plotting
        self.fig_obj, self.ax_obj = plt.subplots(1, 1)
        self.fig_obj.suptitle(name + ": Objective")
        self.ax_obj.set_ylabel(self.statistical_measure)
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

    def calibration_callback_func(self, xk, obj, verbose_information, penalty=None):
        """
        Logs the current values of the objective function.

        :param np.array xk:
            Array with the current values of the calibration
        :param float obj:
            Current objective value.
        :param dict verbose_information:
            A dict with difference-values of for all goals and the
            corresponding weightings
        """
        # Call the logger function to print and log
        super().calibration_callback_func(xk, obj, verbose_information, penalty)
        # Plot the current objective value
        self.ax_obj.plot(self._counter_calibration, obj, "ro")

        # Plot the tuner parameters
        self._plot_tuner_parameters(xk=xk)

        # Plot the measured and simulated data
        if self.goals is not None and self.create_tsd_plot:
            self._plot_goals()

        if self.show_plot:
            plt.draw()
            plt.pause(1e-5)

    def save_calibration_result(self, res, model_name, duration, itercount, **kwargs):
        """
        Process the result, re-run the simulation and generate
        a logFile for the minimal quality measurement

        :param scipy.optimize.minimize.result res:
            Result object of the minimization
        :param str model_name:
            Name of the model being calibrated
        :param str statistical_measure:
            Statistical measure used for calibration.
            One of the supported methods in
            ebcpy.utils.statistics_analyzer.StatisticsAnalyzer
        :keyword str file_type:
            svg, pdf or png
        """
        file_type = "svg"
        if isinstance(kwargs.get("file_type"), str):
            file_type = kwargs.get("file_type")
        super().save_calibration_result(res, model_name, duration, itercount)

        # Extract filepathes
        iterpath = os.path.join(self.cd, f'Iteration_{itercount}')
        if not os.path.exists(iterpath):
            os.mkdir(iterpath)

        filepath_tuner = os.path.join(iterpath, "tuner_parameter_plot.%s" % file_type)
        filepath_obj = os.path.join(iterpath, "objective_plot.%s" % file_type)
        bestgoal = os.path.join(self.cd, self.goals_dir, str(res["Iterate"]) + "_goals.%s" % file_type)

        #filepath_bestgoal = os.path.join(iterpath, str(res["Iterate"]) + "_goals.%s" % file_type)


        # Save figures & close plots
        copyfile(bestgoal, f'{iterpath}\\best_goals.%s' % file_type)
        self.fig_tuner.savefig(filepath_tuner)
        self.fig_obj.savefig(filepath_obj)
        plt.close("all")

        if res['better_current_result'] == True:
            # save improvement of recalibration ("best goals df" as csv)
            res['Goals'].get_goals_data().to_csv(os.path.join(iterpath, 'goals_df.csv'), sep=",", decimal=".")

        # Safe Resultfiles as csv
        # obj & tuner
        cal_results = res
        cal_results['duration'] = duration
        cal_results = data_types.TimeSeriesData(res)
        cal_results.to_csv(f'{self.cd}\\Iteration_{itercount}\\RESULTS_{self.calibration_class.name}_iteration{itercount}.csv')
        # best goals dataframe
        best_goals_df = res['Goals'].get_goals_data()


    def log_intersection_of_tuners(self, intersected_tuner_parameters, itercount):
        """
        If an intersection for multiple classes occurs, an information about
        the statistics of the dataset has to be provided.

        :param dict intersected_tuner_parameters:
            Dict with cols being the name of the tuner parameter and the
            value being the list with all the different "best" values for
            the tuner parameter.
        """
        super().log_intersection_of_tuners(intersected_tuner_parameters, itercount)
        x_labels = intersected_tuner_parameters.keys()
        data = list(intersected_tuner_parameters.values())
        fig_intersection, ax_intersection = plt.subplots(1, len(x_labels), squeeze=False)
        for i, x_label in enumerate(x_labels):
            cur_ax = ax_intersection[0][i]
            cur_ax.violinplot(data[i], showmeans=True, showmedians=False,
                              showextrema=True)
            cur_ax.plot([1] * len(data[i]), data[i], "ro", label="Results")

            cur_ax.get_xaxis().set_tick_params(direction='out')
            cur_ax.xaxis.set_ticks_position('bottom')
            cur_ax.set_xticks(np.arange(1, 2))
            cur_ax.set_xlim(0.25, 1.75)
            cur_ax.set_xticklabels([x_label])
            cur_ax.legend(loc="upper right")

        # Always store in the parent diretory as this info is relevant for all classes
        fig_intersection.suptitle("Intersection of Tuner Parameters")
        path_intersections = os.path.join(os.path.dirname(self.cd), "tunerintersections")
        if not os.path.exists(path_intersections):
            os.makedirs(path_intersections)
        fig_intersection.savefig(os.path.join(path_intersections, f'tuner_parameter_intersection_plot_it{itercount}.svg'))
        if self.show_plot:
            plt.draw()
            plt.pause(15)

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

        # Get information on the relevant-intervals of the calibration:
        rel_intervals = self.calibration_class.relevant_intervals

        _goals_df = self.goals.get_goals_data()
        _goals_names = self.goals.get_goals_list()
        goal_counter = 0
        for row in range(self._n_rows_goals):
            for col in range(self._n_cols_goals):
                cur_ax = self.ax_goal[row][col]
                cur_ax.cla()
                if goal_counter >= self._num_goals:
                    cur_ax.axis("off")
                else:
                    cur_goal = _goals_names[goal_counter]
                    cur_ax.plot(_goals_df[cur_goal, self.goals.sim_tag_str],
                                label=cur_goal + "_{}".format(self.goals.sim_tag_str),
                                linestyle="--", color="r")
                    cur_ax.plot(_goals_df[cur_goal, self.goals.meas_tag_str],
                                label=cur_goal + "_{}".format(self.goals.meas_tag_str),
                                color="b")
                    # Mark the disregarded intervals in grey
                    _start = self.calibration_class.start_time
                    _first = True  # Only create one label
                    for interval in rel_intervals:
                        _end = interval[0]
                        if _first:
                            cur_ax.axvspan(_start, _end, facecolor="grey", alpha=0.7, label="Disregarded Interval")
                            _first = False
                        # Only create one label
                        else:
                            cur_ax.axvspan(_start, _end, facecolor="grey", alpha=0.7)
                        _start = interval[1]
                    # Final plot of grey
                    cur_ax.axvspan(rel_intervals[-1][-1], self.calibration_class.stop_time, facecolor="grey", alpha=0.5)

                    cur_ax.legend(loc="lower right")
                    cur_ax.set_xlabel("Time / s")
                goal_counter += 1

        if self.save_tsd_plot:
            self.goals_dir = "TimeSeriesPlot"
            _savedir = os.path.join(self.cd, self.goals_dir)
            if not os.path.exists(_savedir):
                os.makedirs(_savedir)
            self.fig_goal.savefig(os.path.join(_savedir,
                                               "{}_goals.svg".format(self._counter_calibration)))
