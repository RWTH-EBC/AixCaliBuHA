"""
Module containing a class for
calibrating multiple calibration classes at once.
"""

import os
from typing import List
import numpy as np
from aixcalibuha import CalibrationClass, data_types
from aixcalibuha.calibration import Calibrator


class MultipleClassCalibrator(Calibrator):
    r"""
    Class for calibration of multiple calibration classes.
    When passing multiple classes of the same name, all names
    are merged into one class with so called relevant time intervals.
    These time intervals are used for the evaluation of the objective
    function. Please have a look at the file in docs\img\typeOfContinouusCalibration.pdf
    for a better understanding on how this class works.

    :param str start_time_method:
        Default is 'fixstart'. Method you want to use to
        specify the start time of your simulation. If 'fixstart' is selected,
        the keyword argument fixstart is used for all classes (Default is 0).
        If 'timedelta' is used, the keyword argument timedelta specifies the
        time being subtracted from each start time of each calibration class.
        Please have a look at the file in docs\img\typeOfContinouusCalibration.pdf
        for a better visualization.
    :param str calibration_strategy:
        Default is 'parallel'. Strategy you want to use for multi-class calibration.
        If 'parallel' is used, parameters will be calibrated on the respective time intervals
        independently. If 'sequential' is used, the order of the calibration classes matters:
        The resulting parameter values of one class will be used as starting values for calibration
        on the next class.
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

    def __init__(self,
                 cd: str,
                 sim_api,
                 calibration_classes: List[CalibrationClass],
                 start_time_method: str = 'fixstart',
                 calibration_strategy: str = 'parallel',
                 **kwargs):
        # Check if input is correct
        if not isinstance(calibration_classes, list):
            raise TypeError("calibration_classes is of type "
                            "%s but should be list" % type(calibration_classes).__name__)

        for cal_class in calibration_classes:
            if not isinstance(cal_class, CalibrationClass):
                raise TypeError(f"calibration_classes is of type {type(cal_class).__name__} "
                                f"but should be CalibrationClass")
        # Pop kwargs of this class (pass parameters and remove from kwarg dict):
        self.merge_multiple_classes = kwargs.pop("merge_multiple_classes", True)
        # Apply (if given) the fix_start_time. Check for correct input as-well.
        self.fix_start_time = kwargs.pop("fix_start_time", 0)
        self.timedelta = kwargs.pop("timedelta", 0)

        # Choose the time-method
        if start_time_method.lower() not in ["fixstart", "timedelta"]:
            raise ValueError(f"Given start_time_method {start_time_method} is not supported. "
                             "Please choose between 'fixstart' or 'timedelta'")
        self.start_time_method = start_time_method

        # Choose the calibration method
        if calibration_strategy.lower() not in ['parallel', 'sequential']:
            raise ValueError(f"Given calibration_strategy {calibration_strategy} is not supported. "
                             f"Please choose between 'parallel' or 'sequential'")
        self.calibration_strategy = calibration_strategy.lower()

        # Instantiate parent-class
        super().__init__(cd, sim_api, calibration_classes[0], **kwargs)
        # Merge the multiple calibration_classes
        if self.merge_multiple_classes:
            self.calibration_classes = data_types.merge_calibration_classes(calibration_classes)
        self._cal_history = []

    def calibrate(self, framework, method=None, **kwargs) -> dict:
        """
        Start the calibration process.

        :return dict self.res_tuner:
            Dictionary of the optimized tuner parameter names and values.
        :return dict self._current_best_iterate:
            Dictionary of the current best results of tuner parameter,
            iteration step, objective value, information
            about the goals object and the penaltyfactor.
        """
        # First check possible intersection of tuner-parameteres
        # and warn the user about it
        all_tuners = []
        for cal_class in self.calibration_classes:
            all_tuners.append(cal_class.tuner_paras.get_names())
        intersection = set(all_tuners[0]).intersection(*all_tuners)
        if intersection and len(self.calibration_classes) > 1:
            self.logger.log("The following tuner-parameters intersect over multiple"
                            f" classes:\n{', '.join(list(intersection))}")

        # Iterate over the different existing classes
        for cal_class in self.calibration_classes:
            #%% Working-Directory:
            # Alter the working directory for saving the simulations-results
            self.cd_of_class = os.path.join(self.cd,
                                            f"{cal_class.name}_"
                                            f"{cal_class.start_time}_"
                                            f"{cal_class.stop_time}")
            self.sim_api.set_cd(self.cd_of_class)

            #%% Calibration-Setup
            # Reset counter for new calibration
            self._counter = 0
            # Retrieve already calibrated parameters (i.e. calibrated in the previous classes)
            already_calibrated_parameters = {}
            for cal_run in self._cal_history:
                for par_name in cal_run['res']['Parameters'].index:
                    already_calibrated_parameters[par_name] = cal_run['res']['Parameters'][par_name]
            # Set fixed names:
            self.fixed_parameters.update(already_calibrated_parameters)

            # Reset best iterate for new class
            self._current_best_iterate = {"Objective": np.inf}
            self.calibration_class = cal_class

            # Set initial values
            initial_values = self.tuner_paras.get_initial_values()
            for idx, par_name in enumerate(self.tuner_paras.get_names()):
                if self.calibration_strategy == "sequential":
                    # Use already calibrated values as initial value for new calibration
                    # Delete it from fixed values and retreive the value
                    initial_values[idx] = self.fixed_parameters.pop(par_name,
                                                                    initial_values[idx])
                else:
                    try:
                        self.fixed_parameters.pop(par_name)  # Just delete, don't use the value
                    except KeyError:
                        pass  # Faster than checking if is in dict.

            self.x0 = self.tuner_paras.scale(initial_values)
            # Either bounds are present or not.
            # If present, the obj will scale the values to 0 and 1. If not
            # we have an unconstrained optimization.
            if self.tuner_paras.bounds is None:
                self.bounds = None
            else:
                self.bounds = [(0, 1) for i in range(len(self.x0))]

            #%% Execution
            # Run the single ModelicaCalibration
            super().calibrate(framework=framework, method=method, **kwargs)

            #%% Post-processing
            # Append result to list for future perturbation based on older results.
            self._cal_history.append({"res": self._current_best_iterate,
                                      "cal_class": cal_class})

        res_tuner = self.check_intersection_of_tuner_parameters()

        # Save calibrated parameter values in JSON
        parameter_values = {}
        for cal_run in self._cal_history:
            for p_name in cal_run['res']['Parameters'].index:
                parameter_values[p_name] = cal_run['res']['Parameters'][p_name]
        self.save_results(parameter_values=parameter_values,
                          filename='MultiClassCalibrationResult')

        return parameter_values

    def _apply_start_time_method(self, start_time):
        """
        Method to be calculate the start_time based on the used
        start-time-method (timedelta or fix-start).

        :param float start_time:
            Start time which was specified by the user in the TOML file.
        :return float start_time - self.timedelta:
            Calculated "timedelta", if specified in the TOML file.
        :return float self.fix_start_time:
            Fix start time which was specified by the user in the TOML file.
        """
        if self.start_time_method == "timedelta":
            # Check if timedelta does not fall below the
            # start_time (start_time should not be lower then zero)
            if start_time - self.timedelta < 0:
                # pylint: disable=import-outside-toplevel
                import warnings
                warnings.warn(
                    'Simulation start time current calibration class \n'
                    ' falls below 0, because of the chosen timedelta. '
                    'The start time will be set to 0 seconds.'
                )
                return 0
            # Using timedelta, _ref_time is subtracted of the given start-time
            return start_time - self.timedelta
        else:
            # With fixed start, the _ref_time parameter is always returned
            return self.fix_start_time

    def check_intersection_of_tuner_parameters(self):
        """
        Checks intersections between tuner parameters.

        :return dict res_tuner:
            Dictionary of the optimized tuner parameter names and values.
        """

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
            self.logger.log_intersection_of_tuners(intersected_tuners,
                                                   itercount=self.recalibration_count)

            # Return average value of ALL tuner parameters (not only intersected).
            # Reason: if there is an intersection of a tuner parameter, but
            # the results of both calibration classes are exactly the same, there
            # is no intersection and the affected parameter will not be
            # delivered to "res_tuner" if one of the other tuners
            # intersect and "intersected_tuners.keys()" is true.
            average_tuner_parameter = {}
            for tuner_para, values in merged_tuner_parameters.items():
                average_tuner_parameter[tuner_para] = sum(values) / len(values)

            self.logger.log("The tuner parameters used for evaluation "
                            "are averaged as follows:\n "
                            "{}".format(' ,'.join([f"{tuner}={value}"
                            for tuner, value in average_tuner_parameter.items()])))

            # Create result-dictionary
            res_tuner = average_tuner_parameter

        return res_tuner
