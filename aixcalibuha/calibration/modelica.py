"""Module containing classes for different
use-cases of calibration, mainly for Modelica
Calibration."""

import os
import numpy as np
from ebcpy import data_types
from aixcalibuha.utils import visualizer
from aixcalibuha.calibration import Calibrator
from aixcalibuha import CalibrationClass


class ModelicaCalibrator(Calibrator):
    """
    Calibrator for Modelica simulation methods. This class can
    be used for single time-intervals of calibration. The objective
    should be the standard-objective function for all calibration
    processes of modelica-models.

    :param str,os.path.normpath cd:
        Working directory
    :param ebcpy.simulationapi.SimulationAPI sim_api:
        Simulation-API for running the models
    :param str statistical_measure:
        Measure to calculate the scalar of the objective,
        e.g. RMSE, MAE, NRMSE
    :param CalibrationClass calibration_class:
        Class with information on Goals and tuner-parameters for calibration
    TODO: Add supported kwargs as a description
    """

    # Dummy variable for accessing the current simulation result
    _filepath_dsres = ""
    # Tuple with information on what time-intervals are relevant for the objective.
    _relevant_time_intervals = []
    # Dummy variable for the result of the calibration:
    _res = None
    # Timedelta before each simulation to initialize the model. Default is zero
    timedelta = 0

    def __init__(self, framework, cd, sim_api, statistical_measure, calibration_class, **kwargs):
        """Instantiate instance attributes"""
        #%% Kwargs
        # Initialize supported keywords with default value
        # Pop the items so they wont be added when calling the
        # __init__ of the parent class.
        self.save_files = kwargs.pop("save_files", False)
        self.timedelta = kwargs.pop("timedelta", 0)

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
        super().__init__(framework, cd, sim_api, statistical_measure, **kwargs)
        if not isinstance(calibration_class, CalibrationClass):
            raise TypeError("calibration_classes is of type {} but should be "
                            "{}".format(type(calibration_class).__name__,
                                        type(CalibrationClass).__name__))
        self.calibration_class = calibration_class
        self.goals = self.calibration_class.goals
        self.tuner_paras = self.calibration_class.tuner_paras
        self.x0 = self.tuner_paras.scale(self.tuner_paras.get_initial_values())
        if self.tuner_paras.bounds is None:
            self.bounds = None
        else:
            # As tuner-parameters are scaled between 0 and 1, the scaled bounds are always 0 and 1
            self.bounds = [(0, 1) for i in range(len(self.x0))]
        # Add the values to the simulation setup.
        self.sim_api.set_sim_setup({"initialNames": self.tuner_paras.get_names(),
                                    "startTime": self.calibration_class.start_time - self.timedelta,
                                    "stopTime": self.calibration_class.stop_time})
        # Set the time-interval for evaluating the objective
        self._relevant_time_intervals = [(self.calibration_class.start_time,
                                          self.calibration_class.stop_time)]

        #%% Setup the logger
        self.logger = visualizer.CalibrationVisualizer(cd, "modelica_calibration",
                                                       self.calibration_class,
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
        if self._relevant_time_intervals:
            # Trim results based on start and end-time
            self.goals.set_relevant_time_intervals(self._relevant_time_intervals)

        #%% Evaluate the current objective
        total_res, unweighted_objective = self.goals.eval_difference(self.statistical_measure,
                                                                     verbose=True)
        if total_res < self._current_best_iterate["Objective"]:
            self._current_best_iterate = {"Iterate": self._counter,
                                          "Objective": total_res,
                                          "Unweighted Objective": unweighted_objective,
                                          "Parameters": xk_descaled,
                                          }

        self.logger.calibration_callback_func(xk, total_res, unweighted_objective)
        return total_res

    def calibrate(self, method=None, framework=None):
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
        self.logger.calibrate_new_class(self.calibration_class)
        self.logger.log_initial_names(self.statistical_measure)

        # Run optimization
        self._res = self.optimize(method, framework)

        #%% Save the relevant results.
        # TODO Make on last simulation to save the result even better!
        self.logger.save_calibration_result(self._current_best_iterate,
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


class MultipleClassCalibrator(ModelicaCalibrator):
    """
    Class for calibration of multiple calibration classes.
    # TODO Add docstrings
    """

    # Default value for the reference time is zero
    reference_start_time = 0

    def __init__(self, framework, cd, sim_api, statistical_measure, calibration_classes,
                 start_time_method='fixstart', reference_start_time=0, **kwargs):
        # Check if input is correct
        if not isinstance(calibration_classes, list):
            raise TypeError("calibration_classes is of type "
                            "%s but should be list" % type(calibration_classes).__name__)

        for cal_class in calibration_classes:
            if not isinstance(cal_class, CalibrationClass):
                raise TypeError("calibration_classes is of type {} but should "
                                "be {}".format(type(cal_class).__name__,
                                               type(CalibrationClass).__name__))

        # Instantiate parent-class
        super().__init__(framework, cd, sim_api, statistical_measure,
                         calibration_classes[0], **kwargs)
        # Merge the multiple calibration_classes
        self.calibration_classes = self._merge_calibration_classes(calibration_classes)
        self._cal_history = []

        # Choose the time-method
        if start_time_method.lower() not in ["fixstart", "timedelta"]:
            raise ValueError("Given start_time_method {} is not supported. Please choose between"
                             "'fixstart' or 'timedelta'".format(start_time_method))
        else:
            self.start_time_method = start_time_method
        # Apply (if given) the reference start time. Check for correct input as-well.
        if not isinstance(reference_start_time, (int, float)):
            raise TypeError("Given reference_start_time is of type {} but "
                            "has to be float or int.".format(type(reference_start_time).__name__))
        self.reference_start_time = reference_start_time

    def calibrate(self, method=None, framework=None):
        # Iterate over the different existing classes
        for cal_class in self.calibration_classes:
            #%% Simulation-Time:
            # Alter the simulation time.
            # The fix-start time or timedelta approach is applied,
            # based on the Boolean in place
            start_time = self._apply_start_time_method(cal_class.start_time)
            self.sim_api.set_sim_setup({"startTime": start_time,
                                        "stopTime": cal_class.stop_time})

            #%% Working-Directory:
            # Alter the working directory for saving the simulations-results
            cd_of_class = os.path.join(self.cd, "{}_{}_{}".format(cal_class.name,
                                                                  cal_class.start_time,
                                                                  cal_class.stop_time))
            self.sim_api.set_cd(cd_of_class)

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
            super().calibrate(method, framework)

            #%% Post-processing
            # Append result to list for future perturbation based on older results.
            self._cal_history.append({"res": self._res,
                                      "cal_class": cal_class})

    def _apply_start_time_method(self, start_time):
        """Method to be calculate the start_time based on the used
        start-time-method (timedelta or fix-start."""
        if self.start_time_method == "timedelta":
            # Using timedelta, _ref_time is subtracted of the given start-time
            return start_time - self.reference_start_time
        else:
            # With fixed start, the _ref_time parameter is always returned
            return self.reference_start_time

    @staticmethod
    def _merge_calibration_classes(calibration_classes):
        # Use a dict for easy name-access
        # TODO: Fetch case were intervals are already in place
        temp_merged = {}
        for cal_class in calibration_classes:
            _name = cal_class.name
            if _name in temp_merged:
                temp_merged[_name]["intervals"].append((cal_class.start_time,
                                                        cal_class.stop_time))
            else:
                temp_merged[_name] = {"goals": cal_class.goals,
                                      "tuner_paras": cal_class.tuner_paras,
                                      "intervals": [(cal_class.start_time,
                                                     cal_class.stop_time)]
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

# TODO: Address plotting and layout issues
# TODO: Create function to create violin plot for intersection of tuner parameters
