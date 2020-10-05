"""Module containing classes for different
use-cases of calibration, mainly for Modelica
Calibration."""

import os
import numpy as np
from ebcpy import data_types
import aixcalibuha
from aixcalibuha.utils import visualizer
from aixcalibuha.calibration import Calibrator
from aixcalibuha import CalibrationClass, Goals
import time


class ModelicaCalibrator(Calibrator):
    """
    Calibrator for Modelica simulation methods. This class can
    be used for single time-intervals of calibration. The objective
    function is the standard-objective function for all calibration
    processes of modelica-models.

    :param WindowsPath cal cd:
        Working directory
    :param ebcpy.simulationapi.SimulationAPI sim_api:
        Simulation-API for running the models
    :param str statistical_measure:
        Measure to calculate the scalar of the objective,
        One of the supported methods in
        ebcpy.utils.statistics_analyzer.StatisticsAnalyzer
        e.g. RMSE, MAE, NRMSE
    :param CalibrationClass calibration_class:
        Class with information on Goals and tuner-parameters for calibration
    :param pd.Dataframe sim_input_data:
        Pandas dataframe of the simulated input data,
        extracted from the specified database (see class GetData in "data_aquisition.py").
    :keyword float timedelta:
        If you use this class for calibrating a single time-interval,
        you can set the timedelta to instantiate the simulation before
        actually evaluating it for the objective.
        The given float (default is 0) is subtracted from the start_time
        of your calibration_class. You can find a visualisation of said timedelta
        in the img folder of the project.
    :keyword boolean save_files:
        If true, all simulation files for each iteration will be saved!
    :keyword boolean verbose_logging:
        Default is True. If False, the standard Logger without
        Visualization in Form of plots is used.
        If you use this, the following keyword arguments below will help
        to further adjust the logging.
    :keyword boolean show_plot:
        Default is True. If False, all created plots are not shown during
        calibration but only stored at the end of the process.
    :keyword boolean create_tsd_plot:
        Default is True. If False, the plot of the time series data (goals)
        is not created and thus shown in during calibration. It therefore is
        also not stored, even if you set the save_tsd_plot keyword-argument to true.
    :keyword boolean save_tsd_plot:
        Default is False. If True, at each iteration the created plot of the
        time-series is saved. This may make the process much slower
    :keyword boolean fail_on_error:
        Default is False. If True, the calibration will stop with an error if
        the simulation fails. See also: ret_val_on_error
    :keyword float,np.NAN ret_val_on_error
        Default is np.NAN. If fail_on_error is false, you can specify here
        which value to return in the case of a failed simulation. Possible
        options are np.NaN, np.inf or some other high numbers. be aware that this
        max influence the solver.
    """

    # Dummy variable for accessing the current simulation result
    _filepath_dsres = ""
    # Tuple with information on what time-intervals are relevant for the objective.
    _relevant_time_intervals = []
    # Dummy variable for the result of the calibration:
    _res = None
    # Timedelta before each simulation to initialize the model. Default is zero
    timedelta = 0
    # Verbose Logging (True: with plots)
    verbose_logging = True
    # Working directory for class
    cd_of_class = None

    def __init__(self, cd, sim_api, statistical_measure, framework, method,
                 calibration_class, sim_input_data, **kwargs):
        """Instantiate instance attributes"""
        #%% Kwargs
        # Initialize supported keywords with default value
        # Pop the items so they wont be added when calling the
        # __init__ of the parent class. Always pop with a default value in case
        # the keyword is not passed.
        self.method = method
        self.framework = framework
        self.verbose_logging = kwargs.pop("verbose_logging", True)
        # self.save_files = kwargs.pop("save_files", False)     # Datensicherung selbst hinterlegt
        # self.timedelta = kwargs.pop("timedelta", 0)
        self.fail_on_error = kwargs.pop("fail_on_error", False)
        self.ret_val_on_error = kwargs.pop("ret_val_on_error", np.NAN)
        self.apply_penalty = kwargs.pop("apply_penalty", True)
        self.sim_input_data = sim_input_data
        # Extract kwargs for the visualizer
        visualizer_kwargs = {"save_tsd_plot": kwargs.pop("save_tsd_plot", None),
                             "create_tsd_plot": kwargs.pop("create_tsd_plot", None),
                             "show_plot": kwargs.pop("show_plot", None),
                             }

        # # Check if types are correct:
        # # Booleans:
        # _bool_kwargs = ["save_files"]
        # for bool_keyword in _bool_kwargs:
        #     keyword_value = self.__getattribute__(bool_keyword)
        #     if not isinstance(keyword_value, bool):
        #         raise TypeError("Given {} is of type {} but should be type "
        #                         "bool".format(bool_keyword,
        #                                       type(keyword_value).__name__))

        #%% Initialize all public parameters
        super().__init__(cd, sim_api, statistical_measure, **kwargs)
        if not isinstance(calibration_class, CalibrationClass):
            raise TypeError("calibration_classes is of type {} but should be "
                            "{}".format(type(calibration_class).__name__,
                                        type(CalibrationClass).__name__))
        self.calibration_class = calibration_class
        self.goals = self.calibration_class.goals
        self.tuner_paras = self.calibration_class.tuner_paras
        # Scale tuner on boundaries
        self.x0 = self.tuner_paras.scale(self.tuner_paras.get_initial_values())
        if self.tuner_paras.bounds is None:
            self.bounds = None
        else:
            # As tuner-parameters are scaled between 0 and 1, the scaled bounds are always 0 and 1
            self.bounds = [(0, 1) for i in range(len(self.x0))]
        if self.start_time_method == "fixstart":
            # reset timedelta if "fixstart" method was chosen
            self.timedelta = 0
        # Add the values to the simulation setup.
        self.sim_api.set_sim_setup({"initialNames": self.tuner_paras.get_names(),
                                    "startTime": self.calibration_class.start_time - self.timedelta,
                                    "stopTime": self.calibration_class.stop_time})
        # Set the time-interval for evaluating the objective
        self._relevant_time_intervals = [(self.calibration_class.start_time,
                                          self.calibration_class.stop_time)]

        #%% Setup the logger
        if self.verbose_logging:
            self.logger = visualizer.CalibrationVisualizer(cd, "modelica_calibration",
                                                           self.calibration_class,
                                                           statistical_measure,
                                                           **visualizer_kwargs)
        else:
            self.logger = visualizer.CalibrationLogger(cd, "modelica_calibration",
                                                       self.calibration_class,
                                                       statistical_measure)

        self.cd_of_class = cd  # Single class does not need an extra folder

    def obj(self, xk, *args):
        """
        Default objective function.
        The usual function will be implemented here:
        1. Convert the set to modelica-units
        2. Simulate the converted-set
        3. Get data as a dataFrame
        4. Calculate the objective based on statistical values

        :param np.array xk:
        Array with normalized values for the minimizer
        :return:
        Objective value based on the used quality measurement
        """
        # edit: This funktion is called by the optimizationframework (scipy, dlib, etc.)
        #%% Initialize class objects
        self._current_iterate = xk
        self._counter += 1
        # Convert tuner set if multiple goals of different scales are used
        xk_descaled = self.tuner_paras.descale(xk)
        # Set initial values for modelica simulation
        self.sim_api.set_initial_values(xk_descaled)
        # Simulate
        try:    # Sichern der Daten selbst hinterlegt
            # Generate the folder name for the calibration
            # if self.save_files:
            #     savepath_files = os.path.join(self.sim_api.cd,
            #                                   "simulation_{}".format(str(self._counter)))
            #     self._filepath_dsres = self.sim_api.simulate(savepath_files=savepath_files)
            #     # %% Load results and write to goals object
            #     sim_target_data = data_types.TimeSeriesData(self._filepath_dsres)
            # else:
            target_sim_names = self.goals.get_sim_var_names()
            self.sim_api.set_sim_setup({"resultNames": target_sim_names})
            # Just specified time intervall in cal classes is simulated
            df = self.sim_api.simulate(self.sim_input_data, savepath_files="")
            # Convert it to time series data object
            sim_target_data = data_types.TimeSeriesData(df)
        except Exception as e:
            if self.fail_on_error:
                raise e
            else:
                return self.ret_val_on_error

        self.goals.set_sim_target_data(sim_target_data)
        if self._relevant_time_intervals:
            # Trim results based on start and end-time of cal class
            self.goals.set_relevant_time_intervals(self._relevant_time_intervals)

        #%% Evaluate the current objective
        # Penalty function (get penaltyfactor)
        if self.sim_api.count > 1 and self.sim_api.benchmark_exists and self.apply_penalty:
            penalty = self.get_penalty(xk_descaled)
            # Evaluate with penalty
            total_res, unweighted_objective = self.goals.eval_difference(self.statistical_measure,
                                                                     verbose=True, penaltyfactor=penalty)
            self.logger.calibration_callback_func(xk, total_res, unweighted_objective, penalty=penalty)

        # There is no benchmark in the first iteration or first iterations were skipped, so no penalty is applied
        else:
            penalty = None
            # Evaluate without penalty
            total_res, unweighted_objective = self.goals.eval_difference(self.statistical_measure,
                                                                     verbose=True)
            self.logger.calibration_callback_func(xk, total_res, unweighted_objective)

        # current best iteration step of current calibration class
        if total_res < self._current_best_iterate["Objective"]:
            #self.best_goals = self.goals
            self._current_best_iterate = {"Iterate": self._counter,
                                          "Objective": total_res,
                                          "Unweighted Objective": unweighted_objective,
                                          "Parameters": xk_descaled,
                                          "Goals": self.goals,
                                          "better_current_result": True,     # For penalty function and for saving goals as csv
                                          "Penaltyfactor": penalty                # Changed to false in this script after calling function "save_calibration_results"
                                          }

        # self.logger.calibration_callback_func(xk, total_res, unweighted_objective, penalty=penalty)
        return total_res

    def calibrate(self):
        #%% Start Calibration:
        self.logger.log("Start calibration of model: {} with "
                        "framework-class {}".format(self.sim_api.model_name,
                                                    self.__class__.__name__))
        self.logger.log("Class: {}, "
                        "Start and Stop-Time of simulation: {}-{} s\n"
                        "Time-Intervals used for "
                        "objective: {}".format(self.calibration_class.name,
                                               self.calibration_class.start_time,
                                               self.calibration_class.stop_time,
                                               self.calibration_class.relevant_intervals))
        # Setup the visualizer for plotting and logging:
        self.logger.calibrate_new_class(self.calibration_class, cd=self.cd_of_class)
        self.logger.log_initial_names()

        # Duration of Calibration
        t_cal_start = time.time()

        # Run optimization
        self._res = self.optimize(self.framework, self.method)

        t_cal_stop = time.time()
        self.t_cal = t_cal_stop - t_cal_start

        #%% Save the relevant results.
        self.logger.save_calibration_result(self._current_best_iterate,
                                            self.sim_api.model_name,
                                            self.t_cal,
                                            self.sim_api.count)
        # Reset
        self._current_best_iterate['better_current_result'] = False

    def validate(self, goals):      # currently not used
        if not isinstance(goals, Goals):
            raise TypeError("Given goals is of type {} but type"
                            "Goals is needed.".format(type(goals).__name__))
        #%% Start Validation:
        self.logger.log("Start validation of model: {} with "
                        "framework-class {}".format(self.sim_api.model_name,
                                                    self.__class__.__name__))
        self.goals = goals
        # Use the results parameter vector to simulate again.
        xk = self._res.x
        val_result = self.obj(xk)
        self.logger.log("{} of validation: {}".format(self.statistical_measure, val_result))

    def _handle_error(self, error):
        """
        Also save the plots if an error occurs.
        See ebcpy.optimization.Optimizer._handle_error for more info.
        """
        self.logger.save_calibration_result(self._current_best_iterate,
                                            self.sim_api.model_name)
        super()._handle_error(error)

    def get_penalty(self, current_tuner_values):
        """
        Get penalty factor for evaluation of current objective. The penaltyfactor
        considers deviations of the tuner parameters in the objective function.
        First the relative deviation between the current best values
        of the tuner parameters from the recalibration steps and
        the tuner parameters obtained in the current iteration step is determined.
        Then the penaltyfactor is being increased according to the relative deviation.

        :param pd.series current_tuner_values:
            To add
        :return: float penalty
            Penaltyfactor for evaluation.
        """
        # TO-DO: Add possibility to consider the sensitivity of tuner parameters

        # Get lists of tuner values (latest best (with all other tuners) & current values)
        previous = self.sim_api.all_tuners_dict
        current = dict(current_tuner_values)
        # Get borders for applying penalty function

        # Apply penalty function
        penalty = 1
        dev_all = []
        for key,value in current.items():
            # Ingore tuner parameter whose current best value is 0
            if previous[key] == 0:
                continue
            # Get relative deviation of tuner values (reference: previous)
            try:
                dev = abs(current[key] - previous[key]) / abs(previous[key])
                # TO-DO: Wie soll Gewichtungsfaktor aussehen für einzelne Tunerparameter bei quadr. Abweichung?
                dev_square = (current[key] - previous[key]) ** 2
            except:
                print('Exception here for Bugfix.')
            # Add corresponding function for penaltyfactor here
            # add 0% to penaltyfactor
            if dev < 0.2:
                continue
            # add 2% to penaltyfactor
            elif dev < 0.4:
                penalty += 0.02
            # add 4% to penaltyfactor
            elif dev < 0.6:
                penalty += 0.04
            # add 8% to penaltyfactor
            else:
                penalty += 0.08
            dev_all.append(dev)

        return penalty



class MultipleClassCalibrator(ModelicaCalibrator):
    """
    Class for calibration of multiple calibration classes.
    When passing multiple classes of the same name, all names
    are merged into one class with so called relevant time intervals.
    These time intervals are used for the evaluation of the objective
    function. Please have a look at the file in \img\typeOfContinouusCalibration.pdf
    for a better understanding on how this class works.

    :param str start_time_method:
        Default is 'fixstart'. Method you want to use to
        specify the start time of your simulation. If 'fixstart' is selected,
        the keyword argument fixstart is used for all classes (Default is 0).
        If 'timedelta' is used, the keyword argument timedelta specifies the
        time being subtracted from each start time of each calibration class.
        Please have a look at the file in \img\typeOfContinouusCalibration.pdf
        for a better visualization.
    :keyword float fix_start_time:
        Value for the fix start time if start_time_method="fixstart". Default is zero.
    :keyword float timedelta:
        Value for timedelta if start_time_method="timedelta". Default is zero.
    :keyword str merge_multiple_classes:
        Default True. If False, the given list of calibration-classes
        is handeled as-is. This means if you pass two CalibrationClass objects
        with the same name (e.g. "device on"), the calibration process will run
        for both these classes stand-alone.
        This will automatically yield an intersection of tuner-parameters, however may
        have advantages in some cases.
    """

    # Default value for the reference time is zero
    fix_start_time = 0
    merge_multiple_classes = True

    def __init__(self, cd, sim_api, statistical_measure, framework, method, calibration_classes, sim_input_data,
                 current_timestamp, start_time_method='fixstart', **kwargs):
        # Check if input is correct (Wird bereits in main gemacht)
        if not isinstance(calibration_classes, list):
            raise TypeError("calibration_classes is of type "
                            "%s but should be list" % type(calibration_classes).__name__)

        for cal_class in calibration_classes:
            if not isinstance(cal_class, CalibrationClass):
                raise TypeError("calibration_classes is of type {} but should "
                                "be {}".format(type(cal_class).__name__,
                                               type(CalibrationClass).__name__))
        # Pop kwargs of this class (pass parameters and remove from kwarg dict):
        self.merge_multiple_classes = kwargs.pop("merge_multiple_classes", True)
        # Apply (if given) the fix_start_time. Check for correct input as-well.
        self.fix_start_time = kwargs.pop("fix_start_time", 0)
        self.timedelta = kwargs.pop("timedelta", 0)
        # Choose the time-method
        if start_time_method.lower() not in ["fixstart", "timedelta"]:
            raise ValueError("Given start_time_method {} is not supported. Please choose between"
                             "'fixstart' or 'timedelta'".format(start_time_method))
        else:
            self.start_time_method = start_time_method

        # Instantiate parent-class
        super().__init__(cd, sim_api, statistical_measure, framework, method,
                         calibration_classes[0], sim_input_data, **kwargs)
        # Merge the multiple calibration_classes
        if self.merge_multiple_classes:
            self.calibration_classes = aixcalibuha.merge_calibration_classes(calibration_classes)
        self._cal_history = []
        # Get current timespamp of calibration
        self.current_timestamp = current_timestamp

    def calibrate(self):

        # Log Start
        self.logger.log("Calibration of day {}, starting at {}. Iterationstep Digital Twin Framework: {}"
                        .format(self.current_timestamp.date(), self.current_timestamp.time(), self.sim_api.count))

        # First check possible intersection of tuner parameters
        # and warn the user about it

        all_tuners = []
        for cal_class in self.calibration_classes:
            all_tuners.append(cal_class.tuner_paras.get_names())
        intersection = set(all_tuners[0]).intersection(*all_tuners)
        if intersection and len(self.calibration_classes) > 1:
            self.logger.log("The following tuner-parameters intersect over multiple"
                            " classes:\n{}".format(", ".join(list(intersection))))

        # Iterate over the different existing classes
        for cal_class in self.calibration_classes:
            #%% Simulation-Time:
            # Alter the simulation time.
            # The fix-start time or timedelta approach is applied
            start_time = self._apply_start_time_method(cal_class.start_time)
            self.sim_api.set_sim_setup({"startTime": start_time,
                                        "stopTime": cal_class.stop_time})

            #%% Working-Directory:
            # Alter the working directory for saving the simulations-results
            self.cd_of_class = os.path.join(self.cd, "{}_{}_{}".format(cal_class.name,
                                                                       cal_class.start_time,
                                                                       cal_class.stop_time))
            self.sim_api.set_cd(self.cd_of_class)

            #%% Calibration-Setup
            # Reset counter for new calibration
            self._counter = 0
            # Reset best iterate for new class
            self._current_best_iterate = {"Objective": np.inf}
            self._relevant_time_intervals = cal_class.relevant_intervals
            self.goals = cal_class.goals
            self.tuner_paras = cal_class.tuner_paras
            self.x0 = self.tuner_paras.scale(self.tuner_paras.get_initial_values())
            # Either bounds are present or not.
            # If present, the obj will scale the values to 0 and 1. If not,
            # we have an unconstrained optimization.
            if self.tuner_paras.bounds is None:
                self.bounds = None
            else:
                self.bounds = [(0, 1) for i in range(len(self.x0))]
            self.sim_api.set_sim_setup({"initialNames": self.tuner_paras.get_names()})
            # Used so that the logger prints the correct class.
            self.calibration_class = cal_class

            #%% Execution
            # Run the single ModelicaCalibration
            super().calibrate()

            #%% Post-processing
            # Append result to list for future perturbation based on older results.
            self._cal_history.append({"res": self._current_best_iterate,
                                      "cal_class": cal_class})

        self.res_tuner = self.check_intersection_of_tuner_parameters()

        # self._current_best_iterate are allways the results from last class which was calibrated
        return self.res_tuner, self._current_best_iterate

    def _apply_start_time_method(self, start_time):
        """Method to be calculate the start_time based on the used
        start-time-method (timedelta or fix-start)."""
        if self.start_time_method == "timedelta":
            # Check if timedelta does not fall below the startime (start_time should not be lower then zero)
            if start_time - self.timedelta < 0:
                import warnings
                warnings.warn('Simulation start time current calibration class \n falls below 0, because'
                              ' of the chosen timedelta. The start time will be set to 0 seconds.'
                              )
                return 0
            else:
                # Using timedelta, _ref_time is subtracted of the given start-time
                return start_time - self.timedelta
        else:
            # With fixed start, the _ref_time parameter is always returned
            return self.fix_start_time

    def check_intersection_of_tuner_parameters(self, prior=True):
        # merge all tuners (writes all values from all classes in one dictionary)
        merged_tuner_parameters = {}
        for cal_class in self._cal_history:
            for tuner_name, best_value in cal_class["res"]["Parameters"].items():
                if (tuner_name in merged_tuner_parameters and
                        best_value not in merged_tuner_parameters[tuner_name]):
                    merged_tuner_parameters[tuner_name].append(best_value)
                else:
                    merged_tuner_parameters[tuner_name] = [best_value]

        # Get tuner parameter
        res_tuner = {}
        for tuner_para, values in merged_tuner_parameters.items():
            res_tuner[tuner_para] = values[0]

        # pop single values, as of no interest
        intersected_tuners = {}
        for tuner_para, values in merged_tuner_parameters.items():
            if len(values) >= 2:
                intersected_tuners[tuner_para] = values

        # Handle tuner intersections
        if intersected_tuners.keys():
            # Plot or log the information, depending on which logger you are using:
            self.logger.log_intersection_of_tuners(intersected_tuners, self.sim_api.count)

            # Return average value of ALL tuner parameters (not only intersected). Reason: if there is an intersection
            # of a tuner parameter, but the results of both calibration classes are exactly the same, there is no
            # intersection and the affected parameter will not be delivered to "res_tuner" if one of the other tuners
            # intersect and "intersected_tuners.keys()" is true.
            average_tuner_parameter = {}
            for tuner_para, values in merged_tuner_parameters.items():
                average_tuner_parameter[tuner_para] = sum(values) / len(values)

            self.logger.log("The tuner parameters used for evaluation are averaged as follows:\n {}"
                            .format(tuner, values) for tuner, values in average_tuner_parameter)

            # Create result-dictonary
            res_tuner = average_tuner_parameter

        return res_tuner