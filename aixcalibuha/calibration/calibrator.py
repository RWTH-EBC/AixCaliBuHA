"""
Module containing the basic class to calibrate
a dynamic model, e.g. a modelica model.
"""

import os
import json
from pathlib import Path
import time
import logging
from typing import Dict, Union
from copy import copy
import numpy as np
import pandas as pd
from ebcpy import data_types, Optimizer
from ebcpy.simulationapi import SimulationAPI
from aixcalibuha.utils import visualizer, MaxIterationsReached, MaxTimeReached
from aixcalibuha import CalibrationClass, Goals, TunerParas


class Calibrator(Optimizer):
    """
    This class can Calibrator be used for single
    time-intervals of calibration.

    :param str,Path working_directory:
        Working directory
    :param ebcpy.simulationapi.SimulationAPI sim_api:
        Simulation-API for running the models
    :param CalibrationClass calibration_class:
        Class with information on Goals and tuner-parameters for calibration
    :keyword str result_path:
        If given, then the resulting parameter values will be stored in a JSON file
        at the given path.
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
        the simulation fails. See also: ``ret_val_on_error``
    :keyword float,np.NAN ret_val_on_error:
        Default is np.NAN. If ``fail_on_error`` is false, you can specify here
        which value to return in the case of a failed simulation. Possible
        options are np.NaN, np.inf or some other high numbers. be aware that this
        max influence the solver.
    :keyword dict fixed_parameters:
        Default is an empty dict. This dict may be used to add certain parameters
        to the simulation which are not tuned / variable during calibration.
        Such parameters may be used if the default values in the model don't
        represent the parameter values you want to use.
    :keyword boolean apply_penalty:
        Default is true. Specifies if a penalty function should be applied or not.
    :keyword boolean penalty_factor:
        Default is 0. Quantifies the impact of the penalty term on the objective function.
        The penalty factor is added to the objective function.
    :keyword boolean recalibration_count:
        Default is 0. Works as a counter and specifies the current cycle of recalibration.
    :keyword boolean perform_square_deviation:
        Default is false.
        If true the penalty function will evaluate the penalty factor with a quadratic approach.
    :keyword int max_itercount:
        Default is Infinity.
        Maximum number of iterations of calibration.
        This may be useful to explicitly limit the calibration
        time.
    :keyword int max_time":
        Deault is Infinity.
        Maximum time in seconds, after which the calibration is stopped. Useful to explicitly limit the calibration time.
    :keyword str plot_file_type:
        File ending of created plots.
        Any supported option in matplotlib, e.g. svg, png, pdf ...
        Default is png

    """

    def __init__(self,
                 working_directory: Union[Path, str],
                 sim_api: SimulationAPI,
                 calibration_class: CalibrationClass,
                 **kwargs):
        """Instantiate instance attributes"""
        # %% Kwargs
        # Initialize supported keywords with default value
        # Pop the items so they wont be added when calling the
        # __init__ of the parent class. Always pop with a default value in case
        # the keyword is not passed.
        self.verbose_logging = kwargs.pop("verbose_logging", True)
        self.save_files = kwargs.pop("save_files", False)
        self.timedelta = kwargs.pop("timedelta", 0)
        self.fail_on_error = kwargs.pop("fail_on_error", False)
        self.ret_val_on_error = kwargs.pop("ret_val_on_error", np.NAN)
        self.fixed_parameters = kwargs.pop("fixed_parameters", {})
        self.apply_penalty = kwargs.pop("apply_penalty", True)
        self.penalty_factor = kwargs.pop("penalty_factor", 0)
        self.recalibration_count = kwargs.pop("recalibration_count", 0)
        self.perform_square_deviation = kwargs.pop("square_deviation", False)
        self.result_path = kwargs.pop('result_path', None)
        self.max_itercount = kwargs.pop('max_itercount', np.inf)
        self.max_time = kwargs.pop('max_time', np.inf)
        self.save_current_best_iterate = kwargs.pop('save_current_best_iterate', False)
        self.at_calibration = True  # Boolean to indicate if validating or calibrating
        # Extract kwargs for the visualizer
        visualizer_kwargs = {
            "save_tsd_plot": kwargs.pop("save_tsd_plot", False),
            "create_tsd_plot": kwargs.pop("create_tsd_plot", True),
            "show_plot": kwargs.pop("show_plot", True),
            "show_plot_pause_time": kwargs.pop("show_plot_pause_time", 1e-3),
            "file_type": kwargs.pop("plot_file_type", "png"),
        }

        # Check if types are correct:
        # Booleans:
        _bool_kwargs = ["save_files"]
        for bool_keyword in _bool_kwargs:
            keyword_value = self.__getattribute__(bool_keyword)
            if not isinstance(keyword_value, bool):
                raise TypeError(f"Given {bool_keyword} is of type "
                                f"{type(keyword_value).__name__} but should be type bool")

        # %% Initialize all public parameters
        super().__init__(working_directory, **kwargs)
        # Set sim_api
        self.sim_api = sim_api

        if not isinstance(calibration_class, CalibrationClass):
            raise TypeError(f"calibration_classes is of type {type(calibration_class).__name__} "
                            f"but should be CalibrationClass")
        self.calibration_class = calibration_class
        # Scale tuner on boundaries
        self.x0 = self.tuner_paras.scale(self.tuner_paras.get_initial_values())
        if self.tuner_paras.bounds is None:
            self.bounds = None
        else:
            # As tuner-parameters are scaled between 0 and 1, the scaled bounds are always 0 and 1
            self.bounds = [(0, 1) for i in range(len(self.x0))]
        # Add the values to the simulation setup.
        self.sim_api.set_sim_setup(
            {"start_time": self.calibration_class.start_time - self.timedelta,
             "stop_time": self.calibration_class.stop_time}
        )

        # %% Setup the logger
        # De-register the logger setup in the optimization class:
        if self.verbose_logging:
            self.logger = visualizer.CalibrationVisualizer(
                working_directory=working_directory,
                name=self.__class__.__name__,
                calibration_class=self.calibration_class,
                logger=self.logger,
                **visualizer_kwargs
            )
        else:
            self.logger = visualizer.CalibrationLogger(
                working_directory=working_directory,
                name=self.__class__.__name__,
                calibration_class=self.calibration_class,
                logger=self.logger
            )

        self.working_directory_of_class = working_directory  # Single class does not need an extra folder

        # Set the output interval according the the given Goals
        mean_freq = self.goals.get_meas_frequency()
        self.logger.log("Setting output_interval of simulation according "
                        f"to measurement target data frequency: {mean_freq}")
        self.sim_api.sim_setup.output_interval = mean_freq
        self.start_time = time.perf_counter()
        
    def _check_for_termination(self):
        if self._counter >= self.max_itercount:
            raise MaxIterationsReached(
                "Terminating calibration as the maximum number "
                f"of iterations {self.max_itercount} has been reached."
            )
            
        if time.perf_counter() - self.start_time > self.max_time:
            raise MaxTimeReached(
                f"Terminating calibration as the maximum time of {self.max_time} s has been "
                f"reached"
            )
        

    def obj(self, xk, *args, verbose: bool = False):
        """
        Default objective function.
        The usual function will be implemented here:

        1. Convert the set to modelica-units
        2. Simulate the converted-set
        3. Get data as a dataFrame
        4. Get penalty factor for the penalty function
        5. Calculate the objective based on statistical values

        :param np.array xk:
            Array with normalized values for the minimizer
        :param int work_id:
            id for worker in Multiprocessing
        :param bool verbose:
            If True, returns the objective value and the unweighted objective dict (for validation).
            If False, returns only the objective value (for optimization).
        :return:
            If verbose == False (default)
                Objective value based on the used quality measurement
            If verbose == True
                Objective value and unweighted objective dict as a tuple
        :rtype: float or tuple
        """
        # Info: This function is called by the optimization framework (scipy, dlib, etc.)
        # Initialize class objects
        self._current_iterate = xk
        self._counter += 1

        # Convert set if multiple goals of different scales are used
        xk_descaled = self.tuner_paras.descale(xk)

        # Set initial values of variable and fixed parameters
        self.sim_api.result_names = self.goals.get_sim_var_names()
        initial_names = self.tuner_paras.get_names()
        parameters = self.fixed_parameters.copy()
        parameters.update({name: value for name, value in zip(initial_names, xk_descaled.values)})
        # Simulate
        # pylint: disable=broad-except
        try:
            # Generate the folder name for the calibration
            if self.save_files:
                savepath_files = os.path.join(self.sim_api.working_directory,
                                              f"simulation_{self._counter}")
                _filepath = self.sim_api.simulate(
                    parameters=parameters,
                    return_option="savepath",
                    savepath=savepath_files,
                    inputs=self.calibration_class.inputs,
                    **self.calibration_class.input_kwargs
                )
                # %% Load results and write to goals object
                sim_target_data = data_types.TimeSeriesData(_filepath)
            else:
                sim_target_data = self.sim_api.simulate(
                    parameters=parameters,
                    inputs=self.calibration_class.inputs,
                    **self.calibration_class.input_kwargs
                )
        except Exception as err:
            if self.fail_on_error:
                self.logger.error("Simulation failed. Raising the error.")
                raise err
            self.logger.error(
                f"Simulation failed. Returning '{self.ret_val_on_error}' "
                f"for the optimization. Error message: {err}"
            )
            return self.ret_val_on_error

        total_res, unweighted_objective = self._kpi_and_logging_calculation(
            xk_descaled=xk_descaled,
            counter=self._counter,
            results=sim_target_data
        )
        self._check_for_termination()
        
        if verbose:
            return total_res, unweighted_objective
        
        return total_res

    def mp_obj(self, x, *args):
        # Initialize list for results
        num_evals = len(x)
        total_res_list = np.empty([num_evals, 1])
        # Set initial values of variable and fixed parameters
        self.sim_api.result_names = self.goals.get_sim_var_names()
        initial_names = self.tuner_paras.get_names()
        parameters = self.fixed_parameters.copy()

        parameter_list = []
        xk_descaled_list = []
        for _xk_single in x:
            # Convert set if multiple goals of different scales are used
            xk_descaled = self.tuner_paras.descale(_xk_single)
            xk_descaled_list.append(xk_descaled)
            # Update Parameters
            parameter_copy = parameters.copy()
            parameter_copy.update(
                {name: value for name, value in zip(initial_names, xk_descaled.values)})
            parameter_list.append(parameter_copy)

        # Simulate
        if self.save_files:
            result_file_names = [f"simulation_{self._counter + idx}" for idx in
                                 range(len(parameter_list))]
            _filepaths = self.sim_api.simulate(
                parameters=parameter_list,
                return_option="savepath",
                savepath=self.sim_api.working_directory,
                result_file_name=result_file_names,
                fail_on_error=self.fail_on_error,
                inputs=self.calibration_class.inputs,
                **self.calibration_class.input_kwargs
            )
            # Load results
            results = []
            for _filepath in _filepaths:
                if _filepath is None:
                    results.append(None)
                else:
                    results.append(data_types.TimeSeriesData(_filepath))
        else:
            results = self.sim_api.simulate(
                parameters=parameter_list,
                inputs=self.calibration_class.inputs,
                fail_on_error=self.fail_on_error,
                **self.calibration_class.input_kwargs
            )

        for idx, result in enumerate(results):
            self._counter += 1
            self._current_iterate = result
            if result is None:
                total_res_list[idx] = self.ret_val_on_error
                continue
            total_res, unweighted_objective = self._kpi_and_logging_calculation(
                xk_descaled=xk_descaled_list[idx],
                counter=self._counter,
                results=result
            )
            # Add single objective to objective list of total Population
            total_res_list[idx] = total_res
        self._check_for_termination()
        
        return total_res_list

    def _kpi_and_logging_calculation(self, *, xk_descaled, counter, results):
        """
        Function to calculate everything needed in the obj or mp_obj
        function after the simulation finished.

        """
        xk = self.tuner_paras.scale(xk_descaled)

        self.goals.set_sim_target_data(results)
        # Trim results based on start and end-time of cal class
        self.goals.set_relevant_time_intervals(self.calibration_class.relevant_intervals)

        # %% Evaluate the current objective
        # Penalty function (get penalty factor)
        if self.recalibration_count > 1 and self.apply_penalty:
            # There is no benchmark in the first iteration or
            # first iterations were skipped, so no penalty is applied
            penaltyfactor = self.get_penalty(xk_descaled, xk)
            # Evaluate with penalty
            penalty = penaltyfactor
        else:
            # Evaluate without penalty
            penaltyfactor = 1
            penalty = None
        total_res, unweighted_objective = self.goals.eval_difference(
            verbose=True,
            penaltyfactor=penaltyfactor
        )
        if self.at_calibration:  # Only plot if at_calibration
            self.logger.calibration_callback_func(
                xk=xk,
                obj=total_res,
                verbose_information=unweighted_objective,
                penalty=penalty
            )
        # current best iteration step of current calibration class
        if total_res < self._current_best_iterate["Objective"]:
            # self.best_goals = self.goals
            self._current_best_iterate = {
                "Iterate": counter,
                "Objective": total_res,
                "Unweighted Objective": unweighted_objective,
                "Parameters": xk_descaled,
                "Goals": self.goals,
                # For penalty function and for saving goals as csv
                "better_current_result": True,
                # Changed to false in this script after calling function "save_calibration_results"
                "Penaltyfactor": penalty
            }
            if self.save_current_best_iterate:
                parameter_values = self._get_parameter_dict_from_current_best_iterate()
                
                temp_save = {
                    "parameters": parameter_values,
                    "objective": total_res
                }
                with open(self.working_directory.joinpath('best_iterate.json'), 'w') as json_file:
                    json.dump(temp_save, json_file, indent=4)

        return total_res, unweighted_objective

    def calibrate(self, framework, method=None, **kwargs) -> dict:
        """
        Start the calibration process of the calibration classes, visualize and save the results.

        The arguments of this function are equal to the
        arguments in Optimizer.optimize(). Look at the docstring
        in ebcpy to know which options are available.
        """
        # %% Start Calibration:
        self.at_calibration = True
        self.logger.log(f"Start calibration of model: {self.sim_api.model_name}"
                        f" with framework-class {self.__class__.__name__}")
        self.logger.log(f"Class: {self.calibration_class.name}, Start and Stop-Time "
                        f"of simulation: {self.calibration_class.start_time}"
                        f"-{self.calibration_class.stop_time} s\n Time-Intervals used"
                        f" for objective: {self.calibration_class.relevant_intervals}")

        # Setup the visualizer for plotting and logging:
        self.logger.calibrate_new_class(self.calibration_class,
                                        working_directory=self.working_directory_of_class,
                                        for_validation=False)
        self.logger.log_initial_names()

        # Duration of Calibration
        t_cal_start = time.time()

        # Run optimization
        try:
            _res = self.optimize(
                framework=framework,
                method=method,
                n_cpu=self.sim_api.n_cpu,
                **kwargs)
        except (MaxIterationsReached, MaxTimeReached) as err:
            self.logger.log(msg=str(err), level=logging.WARNING)
        t_cal_stop = time.time()
        t_cal = t_cal_stop - t_cal_start

        # Check if optimization worked correctly
        if "Iterate" not in self._current_best_iterate:
            raise Exception(
                "Some error during calibration yielded no successful iteration. "
                "Can't save or return any results."
            )

        # %% Save the relevant results.
        self.logger.save_calibration_result(self._current_best_iterate,
                                            self.sim_api.model_name,
                                            duration=t_cal,
                                            itercount=self.recalibration_count)
        # Reset
        self._current_best_iterate['better_current_result'] = False

        # Save calibrated parameter values in JSON
        parameter_values = self._get_parameter_dict_from_current_best_iterate()
        self.save_results(parameter_values=parameter_values,
                          filename=self.calibration_class.name)
        return parameter_values
    
    def _get_parameter_dict_from_current_best_iterate(self) -> dict:
        """
        Get the parameter dictionary from the current best iterate.
        """
        parameter_values = {}
        for p_name in self._current_best_iterate['Parameters'].index:
            parameter_values[p_name] = self._current_best_iterate['Parameters'][p_name]
        return parameter_values

    @property
    def calibration_class(self) -> CalibrationClass:
        """Get the current calibration class"""
        return self._cal_class

    @calibration_class.setter
    def calibration_class(self, calibration_class: CalibrationClass):
        """Set the current calibration class"""
        self.sim_api.set_sim_setup(
            {"start_time": self._apply_start_time_method(start_time=calibration_class.start_time),
             "stop_time": calibration_class.stop_time}
        )
        self._cal_class = calibration_class

    @property
    def tuner_paras(self) -> TunerParas:
        """Get the current tuner parameters of the calibration class"""
        return self.calibration_class.tuner_paras

    @tuner_paras.setter
    def tuner_paras(self, tuner_paras: TunerParas):
        """Set the current tuner parameters of the calibration class"""
        self.calibration_class.tuner_paras = tuner_paras

    @property
    def goals(self) -> Goals:
        """Get the current goals of the calibration class"""
        return self.calibration_class.goals

    @goals.setter
    def goals(self, goals: Goals):
        """Set the current goals of the calibration class"""
        self.calibration_class.goals = goals

    @property
    def fixed_parameters(self) -> dict:
        """Get the currently fixed parameters during calibration"""
        return self._fixed_pars

    @fixed_parameters.setter
    def fixed_parameters(self, fixed_parameters: dict):
        """Set the currently fixed parameters during calibration"""
        self._fixed_pars = fixed_parameters

    def save_results(self, parameter_values: dict, filename: str):
        """Saves the given dict into a file with path
        self.result_path and name filename."""
        if self.result_path is not None:
            os.makedirs(self.result_path, exist_ok=True)
            s_path = os.path.join(self.result_path, f'{filename}.json')
            with open(s_path, 'w') as json_file:
                json.dump(parameter_values, json_file, indent=4)

    def validate(self, validation_class: CalibrationClass, calibration_result: Dict, verbose=False):
        """
        Validate the given calibration class based on the given
        values for tuner_parameters.

        :param CalibrationClass validation_class:
            The class to validate on
        :param dict calibration_result:
            The calibration result to apply to the validation class on.
        """
        # Start Validation:
        self.at_calibration = False
        self.logger.log(f"Start validation of model: {self.sim_api.model_name} with "
                        f"framework-class {self.__class__.__name__}")
        # Use start-time of calibration class
        self.calibration_class = validation_class
        start_time = self._apply_start_time_method(
            start_time=self.calibration_class.start_time
        )
        old_tuner_paras = copy(self.calibration_class.tuner_paras)
        tuner_values = list(calibration_result.values())
        self.calibration_class.tuner_paras = TunerParas(
            names=list(calibration_result.keys()),
            initial_values=tuner_values,
            # Dummy bounds as they are scaled anyway
            bounds=[(val - 1, val + 1) for val in tuner_values]
        )

        # Set the start-time for the simulation
        self.sim_api.sim_setup.start_time = start_time

        self.logger.calibrate_new_class(self.calibration_class,
                                        working_directory=self.working_directory_of_class,
                                        for_validation=True)

        # Use the results parameter vector to simulate again.
        self._counter = 0  # Reset to one
        # Scale the tuner parameters
        xk = self.tuner_paras.scale(tuner_values)
        # Evaluate objective
        obj, unweighted_objective = self.obj(xk=xk, verbose=True)
        self.logger.validation_callback_func(
            obj=obj
        )
        # Reset tuner_parameters to avoid unwanted behaviour
        self.calibration_class.tuner_paras = old_tuner_paras
        if verbose:
            weights = [1]
            objectives = [obj]
            goals = ['all']
            for goal, val in unweighted_objective.items():
                weights.append(val[0])
                objectives.append(val[1])
                goals.append(goal)
            index = pd.MultiIndex.from_product(
                [[validation_class.name], goals],
                names=['Class', 'Goal']
            )
            obj_verbos = pd.DataFrame(
                {'weight': weights, validation_class.goals.statistical_measure: objectives},
                index=index
            )
            return obj_verbos
        return obj

    def _handle_error(self, error):
        """
        Also save the plots if an error occurs.
        See ebcpy.optimization.Optimizer._handle_error for more info.
        """
        # This error is our own, we handle it in the calibrate() function
        if isinstance(error, (MaxIterationsReached, MaxTimeReached)):
            raise error
        self.logger.save_calibration_result(best_iterate=self._current_best_iterate,
                                            model_name=self.sim_api.model_name,
                                            duration=0,
                                            itercount=0)
        super()._handle_error(error)

    def get_penalty(self, current_tuners, current_tuners_scaled):
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
        previous_scaled = self.sim_api.all_tuners_dict_scaled
        # previous_scaled = list(self.sim_api.all_tuners_dict.keys())
        current = current_tuners
        current_scaled = dict(current_tuners_scaled)

        # Apply penalty function
        penalty = 1
        for key, value in current_scaled.items():
            # Add corresponding function for penaltyfactor here
            if self.perform_square_deviation:
                # Apply quadratic deviation
                dev_square = (value - previous_scaled[key]) ** 2
                penalty += self.penalty_factor * dev_square
            else:
                # Apply relative deviation
                # Ingore tuner parameter whose current best value is 0
                if previous[key] == 0:
                    continue
                # Get relative deviation of tuner values (reference: previous)
                try:
                    dev = abs(current[key] - previous[key]) / abs(previous[key])
                    penalty += self.penalty_factor * dev
                except ZeroDivisionError:
                    pass

        return penalty

    def _apply_start_time_method(self, start_time):
        """
        Method to be calculate the start_time based on the used
        timedelta method.

        :param float start_time:
            Start time which was specified by the user in the TOML file.
        :return float start_time - self.timedelta:
            Calculated "timedelta", if specified in the TOML file.
        """
        return start_time - self.timedelta
