"""Module containing classes for different
use-cases of calibration, mainly for Modelica
Calibration."""

import os
from abc import abstractmethod
from aixcal import simulationapi
from aixcal import data_types
from aixcal.utils import visualizer
from aixcal.optimizer import Calibrator
import modelicares.simres as sr
import modelicares.util as mrutil


class ModelicaCalibrator(Calibrator):
    """
    Calibrator for Modelica simulation methods. This class can
    be used for single time-intervals of calibration. The objective
    should be the standard-objective function for all calibration
    processes of modelica-models.

    :param str,os.path.normpath cd:
        Working directory
    :param aixcal.simulationapi.SimulationAPI sim_api:
        Simulation-API for running the models
    :param str statistical_measure:
        Measure to calculate the scalar of the objective,
        e.g. RMSE, MAE, NRMSE
    :param CalibrationClass calibration_class:
        Class with information on Goals and tuner-parameters for calibration
    """

    # Dummy variable for accessing the current simulation result
    _filepath_dsres = ""
    # Tuple with information on what time-intervals are relevant for the objective.
    _relevant_time_interval = ()
    # Dummy variable for the result of the calibration:
    _res = None

    def __init__(self, cd, sim_api, statistical_measure, calibration_class, **kwargs):
        """Instantiate instance attributes"""
        #%% Kwargs
        # Initialize supported keywords with default value
        self.save_files = kwargs.pop("save_files", False)

        # Check if types are correct:
        # Booleans:
        _bool_kwargs = ["save_files"]
        for bool_keyword in _bool_kwargs:
            keyword_value = self.__getattribute__(bool_keyword)
            if not isinstance(keyword_value, bool):
                raise TypeError("Given {} is of type {} but should be type "
                                "bool".format(bool_keyword,
                                              type(keyword_value).__name__))

        #%% Initialize all public parameters
        super().__init__(cd, sim_api, statistical_measure, **kwargs)
        if not isinstance(calibration_class, data_types.CalibrationClass):
            raise TypeError("calibration_classes is of type {} but should be "
                            "{}".format(type(calibration_class).__name__,
                                        type(data_types.CalibrationClass).__name__))
        self.calibration_class = calibration_class
        self.goals = self.calibration_class.goals
        self.tuner_paras = self.calibration_class.tuner_paras
        self.x0 = self.tuner_paras.scale(self.tuner_paras.get_initial_values())
        if self.tuner_paras.bounds is None:
            self.bounds = None
        else:
            self.bounds = [(0, 1) for i in range(len(self.x0))]
        # Add the values to the simulation setup.
        self.sim_api.set_sim_setup({"initialNames": self.tuner_paras.get_names(),
                                    "startTime": self.calibration_class.start_time,
                                    "stopTime": self.calibration_class.stop_time})

        #%% Setup the logger
        self.logger = visualizer.CalibrationVisualizer(cd, "modelica_calibration",
                                                       self.tuner_paras,
                                                       self.goals,
                                                       show_plot=self.show_plot)

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
        #%% Initialize class objects
        self._current_iterate = xk
        self._counter += 1
        # Convert set if multiple goals of different scales are used
        xk_descaled = self.tuner_paras.descale(xk)
        # Set initial values for modelica simulation
        self.sim_api.set_initial_values(xk_descaled)
        # Generate the folder name for the calibration
        if self.save_files:
            savepath_files = os.path.join(self.sim_api.cd,
                                          "simulation_{}".format(str(self._counter)))
        else:
            savepath_files = ""
        # Simulate
        self._filepath_dsres = self.sim_api.simulate(savepath_files=savepath_files)
        #%% Load results and write to goals object
        sim_target_data = data_types.SimTargetData(self._filepath_dsres)
        self.goals.set_sim_target_data(sim_target_data)
        if self._relevant_time_interval:
            # Trim results based on start and end-time
            self.goals.set_relevant_time_interval(self._relevant_time_interval[0],
                                                  self._relevant_time_interval[1])

        #%% Evaluate the current objective
        total_res = self.goals.eval_difference(self.statistical_measure)
        self._obj_his.append(total_res)
        self.logger.calibration_callback_func(xk, total_res)
        return total_res

    def run(self, method, framework):
        #%% Setup the method and framework in use
        super().run(method, framework)

        #%% Start Calibration:
        self.logger.log("Start calibration of model: {} with "
                        "framework-class {}".format(self.sim_api.model_name,
                                                    self.__class__.__name__))
        self.logger.log("Class: {}, "
                        "Time-Interval: {}-{} s".format(self.calibration_class.name,
                                                        self.calibration_class.start_time,
                                                        self.calibration_class.stop_time))
        self.logger.log_initial_names(self.statistical_measure)
        # Setup the visualizer for plotting:
        self.logger.calibrate_new_class(self.calibration_class.name, self.tuner_paras, self.goals)
        # Run optimization
        self._res = self._minimize_func(method)

        #%% Save the relevant results.
        self.logger.save_calibration_result(self._res,
                                            self.sim_api.model_name,
                                            self.statistical_measure)

    def validate(self, goals):
        if not isinstance(goals, data_types.Goals):
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


class ContinuousModelicaCalibration(ModelicaCalibrator):
    """
    Base-Class for continuous calibration of multiple calibration
    classes.
    """

    def __init__(self, cd, sim_api, statistical_measure, calibration_classes, **kwargs):
        if not isinstance(calibration_classes, list):
            raise TypeError("calibration_classes is of type "
                            "%s but should be list" % type(calibration_classes).__name__)
        for cal_class in calibration_classes:
            if not isinstance(cal_class, data_types.CalibrationClass):
                raise TypeError("calibration_classes is of type {} but should "
                                "be {}".format(type(cal_class).__name__,
                                               type(data_types.CalibrationClass).__name__))
        super().__init__(cd, sim_api, statistical_measure, calibration_classes[0], **kwargs)
        self.calibration_classes = calibration_classes
        self._cal_history = []

    def run(self, method, framework):
        curr_num = 0
        for cal_class in self.calibration_classes:
            #%% Simulation-Time:
            # Alter the simulation time. This depends on the mode one is using.
            # The method ref_start_time will be overloaded by children of this class.
            start_time = self.ref_start_time(cal_class.start_time)
            self.sim_api.set_sim_setup({"startTime": start_time,
                                        "stopTime": cal_class.stop_time})

            #%% Working-Directory:
            # Alter the working directory for the simulations
            cd_of_class = os.path.join(self.cd, "{}_{}".format(curr_num, cal_class.name))
            self.sim_api.set_cd(cd_of_class)

            #%% Tuner-Parameters
            # Alter tunerParas based on old results
            if self._cal_history:
                self.process_in_between_classes(cal_class.tuner_paras)
            else:
                self.tuner_paras = cal_class.tuner_paras

            #%% Calibration-Setup
            # Reset counter for new calibration
            self._counter = 0
            self._relevant_time_interval = (cal_class.start_time, cal_class.stop_time)
            self.goals = cal_class.goals
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
            super().run(method, framework)

            #%% Post-processing
            # Append result to list for future perturbation based on older results.
            self._cal_history.append({"tuner_paras": self.tuner_paras,
                                      "res": self._res,
                                      "cal_class": cal_class})
            curr_num += 1
            self._counter = 0  # Reset counter for next optimization

    @abstractmethod
    def ref_start_time(self, start_time):
        """Method to be calculate the start_time based on the used
        way of continuous calibration."""
        raise NotImplementedError('{}.ref_start_time function is '
                                  'not defined'.format(self.__class__.__name__))

    def process_in_between_classes(self, tuner_paras_of_class):
        """Method to execute some function between the calibration
        of two classes. The basic step is to alter the tuner paramters
        based on past-optimal values.

        :param data_types.TunerParas tuner_paras_of_class:
            TunerParas of the next class.
        """
        self.tuner_paras = self._alter_tuner_paras(tuner_paras_of_class)

    def _alter_tuner_paras(self, tuner_paras_of_class):
        """
        Based on old calibration results, this function
        alters the start-values for the new tuner_paras-Set

        :param aixcal.data_types.TunerParas tuner_paras_of_class:
            Tuner Parameters for the next time-interval
        :return: aixcal.data_types.TunerParas tunerParaDict:
            TunerParas with the altered values
        """
        total_time = 0
        relevant_tuner_paras = tuner_paras_of_class.get_names()
        average_ini_vals = {}
        for cal_his in self._cal_history:
            tuner_paras = cal_his["cal_class"].tuner_paras
            tuner_paras_opt = tuner_paras.scale(cal_his["res"].x)
            timedelta = cal_his["cal_class"].stop_time - cal_his["cal_class"].start_time
            total_time += timedelta
            for i in range(0, len(tuner_paras.get_names())):
                name = tuner_paras.get_names()[i]
                if name in average_ini_vals.keys():
                    average_ini_vals[name] += tuner_paras_opt[i] * timedelta
                else:
                    average_ini_vals[name] = tuner_paras_opt[i] * timedelta
        for name, ini_val in average_ini_vals.items():
            average_ini_vals[name] = ini_val / total_time  # Build average again
        # Alter the values of the tuner_paras
        for name in list(set(average_ini_vals.keys()).intersection(relevant_tuner_paras)):
            tuner_paras_of_class.set_value(name, "initial_value", average_ini_vals[name])

        return tuner_paras_of_class


class FixStartContModelicaCal(ContinuousModelicaCalibration):
    """Class for continuous calibration using a fixed start.
    All simulations will start at that fixed point no matter the
    actual start-time of the calibration-class. For obj-values, only
    the relevant interval is taken into account"""

    def __init__(self, cd, sim_api, statistical_measure,
                 calibration_classes, fix_start_time, **kwargs):
        super().__init__(cd, sim_api, statistical_measure, calibration_classes, **kwargs)
        self._fix_start_time = fix_start_time

    def ref_start_time(self, start_time):
        return self._fix_start_time


class TimedeltaContModelicaCal(ContinuousModelicaCalibration):
    """Class for continuous calibration using a fixed timedelta.
    Before each calibration-class starts, a fix timedelta is
    subtracted from the start-time to ensure the simulation
    is at a steady-point for the calibration."""

    def __init__(self, cd, sim_api, statistical_measure,
                 calibration_classes, timedelta, **kwargs):
        super().__init__(cd, sim_api, statistical_measure, calibration_classes, **kwargs)
        self._timedelta = timedelta

    def ref_start_time(self, start_time):
        return start_time - self._timedelta


class DsFinalContModelicaCal(ContinuousModelicaCalibration):
    """
    Class for continuous calibration using the dsfinal.txt.file
    after each calibration-class to import the inital values based
    on the final values of the last class.
    """

    def __init__(self, cd, sim_api, statistical_measure,
                 calibration_classes, **kwargs):
        super().__init__(cd, sim_api, statistical_measure, calibration_classes, **kwargs)
        self._total_min_dsfinal_path = os.path.join(cd, "total_min_dsfinal", "dsfinal.txt")
        # For calibration of multiple classes, the dsfinal is of interest.
        self._total_min = 1e308
        self._total_initial_names = self._join_tuner_paras()
        self._traj_names = []
        # Savepath where the mat-file and the dsfinal of the current best iterate is saved.

    def obj(self, xk, *args):
        # Evaluate the objective
        obj_val = super().obj(xk, *args)
        # Get all trajectory names of the dsres-file
        sim = sr.SimRes(self._filepath_dsres)
        self._traj_names = sim.get_trajectories()
        # Get the current best dsfinal-file for next calibration-interval.
        if obj_val < self._total_min:
            self._total_min = obj_val
            # Overwrite old results:
            if os.path.isfile(self._total_min_dsfinal_path):
                os.remove(self._total_min_dsfinal_path)
            os.rename(os.path.join(os.path.dirname(self._filepath_dsres), "dsfinal.txt"),
                      self._total_min_dsfinal_path)
        return obj_val

    def ref_start_time(self, start_time):
        return start_time

    def process_in_between_classes(self, tuner_paras_of_class):
        super().process_in_between_classes(tuner_paras_of_class)
        # Alter the dsfinal for the new phase
        new_dsfinal = os.path.join(self.sim_api.cwdir, "dsfinal.txt")
        self._total_initial_names = list(set(self._total_initial_names + self._traj_names))
        mrutil.eliminate_parameters_from_ds_file(self._total_min_dsfinal_path,
                                                 new_dsfinal,
                                                 self._total_initial_names)
        self.sim_api.import_initial(new_dsfinal)

    def _join_tuner_paras(self):
        """
        Join all initialNames used for calibration in the given dataset. This function
        is used to find all values to be filtered in the dsfinal-file.

        :return: list
        Joined list of all names.
        """
        joined_list = []
        for cal_class in self.calibration_classes:
            joined_list.append(cal_class.tuner_paras.get_names())
        return list(set(joined_list))
