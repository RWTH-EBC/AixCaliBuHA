"""Package containing modules for sensitivity analysis.
The module contains the relevant base-classes."""
import abc
import copy
import os
import time
from typing import List
import numpy as np
import pandas as pd
from ebcpy.utils import setup_logger
from ebcpy import data_types
from ebcpy.simulationapi import SimulationAPI
from aixcalibuha import CalibrationClass, data_types
from aixcalibuha import utils


class SenAnalyzer(abc.ABC):
    """
    Class to perform a Sensitivity Analysis.

    :param SimulationAPI sim_api:
        Simulation-API used to simulate the samples
    :param int num_samples:
        The parameter `N` to the sampler methods of sobol and morris. NOTE: This is not the
        the number of samples produced, but relates to the total number of samples produced in
        a manner dependent on the sampler method used. See the documentation of sobol and
        morris in the SALib for more information.
    :keyword str analysis_function:
        Used to automatically select result values.
    :keyword str,os.path.normpath cd:
        The path for the current working directory.
    :keyword boolean fail_on_error:
        Default is False. If True, the calibration will stop with an error if
        the simulation fails. See also: ``ret_val_on_error``
    :keyword float,np.NAN ret_val_on_error:
        Default is np.NAN. If ``fail_on_error`` is false, you can specify here
        which value to return in the case of a failed simulation. Possible
        options are np.NaN, np.inf or some other high numbers. be aware that this
        max influence the solver.
    :keyword boolean save_files:
        If true, all simulation files for each iteration will be saved!

    """

    def __init__(self,
                 sim_api: SimulationAPI,
                 num_samples: int,
                 **kwargs):
        """Instantiate class parameters"""
        # Setup the instance attributes
        self.sim_api = sim_api
        self.num_samples = num_samples

        # Update kwargs
        self.fail_on_error = kwargs.pop("fail_on_error", True)
        self.save_files = kwargs.pop("save_files", False)
        self.ret_val_on_error = kwargs.pop("ret_val_on_error", np.NAN)
        self.cd = kwargs.pop("cd", os.getcwd())
        self.analysis_variable = kwargs.pop('analysis_variable',
                                            self.analysis_variables[0])
        if self.analysis_variable not in self.analysis_variables:
            raise TypeError(f'Given analysis_variable "{self.analysis_variable}" not '
                            f'supported for class {self.__class__.__name__}. '
                            f'Supported options are: {", ".join(self.analysis_variables)}.')

        # Setup the logger
        self.logger = setup_logger(cd=self.cd, name=self.__class__.__name__)

        # Setup default values
        self.problem: dict = None

    @property
    @abc.abstractmethod
    def analysis_variables(self) -> List[str]:
        """
        Indicate which variables are
        able to be selected for analysis

        :return:
            A list of strings
        :rtype: List[str]
        """
        raise NotImplementedError(f'{self.__class__.__name__}.analysis_variables '
                                  f'property is not defined yet')

    @abc.abstractmethod
    def analysis_function(self, x, y):
        """
        Use the method to analyze the simulation results.

        :param np.array x:
            the `X` parameter of the method (The NumPy matrix containing the model inputs)
        :param np.array y:
            The NumPy array containing the model outputs
        """
        raise NotImplementedError(f'{self.__class__.__name__}.analysis_function '
                                  f'function is not defined yet')

    @abc.abstractmethod
    def create_sampler_demand(self) -> dict:
        """
        Return the sampler parameters

        :return:
            dict: A dict with the sampler demand
        """
        raise NotImplementedError(f'{self.__class__.__name__}.analysis_function '
                                  f'function is not defined yet')

    @abc.abstractmethod
    def generate_samples(self):
        """
        Run the sampler specified by `method` and return the results.

        :return:
            The list of samples generated as a NumPy array with one row per sample
            and each row containing one value for each variable name in `problem['names']`.
        :rtype: np.ndarray
        """
        raise NotImplementedError(f'{self.__class__.__name__}.generate_samples '
                                  f'function is not defined yet')

    def simulate_samples(self, samples, cal_class):
        """
        Put the parameters in the model and simulate it.

        :param Union[list, np.ndarray] samples:
            Output variables in dymola
        :param cal_class:
            One class for calibration. Goals and tuner_paras have to be set

        :returns: np.array
            An array containing the evaluated differences for each sample
        """
        output = []
        # Set the output interval according the the given Goals
        mean_freq = cal_class.goals.get_meas_frequency()
        self.logger.info("Setting output_interval of simulation according "
                         "to measurement target data frequency: %s", mean_freq)
        self.sim_api.sim_setup.output_interval = mean_freq
        initial_names = cal_class.tuner_paras.get_names()
        self.sim_api.set_sim_setup({"start_time": cal_class.start_time,
                                    "stop_time": cal_class.stop_time})
        self.sim_api.result_names = cal_class.goals.get_sim_var_names()
        for i, initial_values in enumerate(samples):
            # Simulate the current values
            self.logger.info('Parameter variation %s of %s',
                             i+1, len(samples))
            parameters = {name: value for name, value in zip(initial_names, initial_values)}
            # Simulate
            # pylint: disable=broad-except
            try:
                # Generate the folder name for the calibration
                if self.save_files:
                    savepath_files = os.path.join(self.cd,
                                                  f"simulation_{i + 1}")
                    filepath = self.sim_api.simulate(
                        parameters=parameters,
                        return_option="savepath",
                        savepath=savepath_files,
                        inputs=cal_class.inputs,
                        **cal_class.input_kwargs
                    )
                    # Load the result file to the goals object
                    sim_target_data = data_types.TimeSeriesData(filepath)
                else:
                    sim_target_data = self.sim_api.simulate(
                        parameters=parameters,
                        inputs=cal_class.inputs,
                        **cal_class.input_kwargs
                    )
            except Exception as err:
                if self.fail_on_error:
                    raise err
                return self.ret_val_on_error

            cal_class.goals.set_sim_target_data(sim_target_data)
            cal_class.goals.set_relevant_time_intervals(cal_class.relevant_intervals)

            # Evaluate the current objective
            total_res = cal_class.goals.eval_difference()
            output.append(total_res)

        return np.asarray(output)

    def run(self, calibration_classes, merge_multiple_classes=True):
        """
        Execute the sensitivity analysis for each class and
        return the result.

        :param CalibrationClass,list calibration_classes:
            Either one or multiple classes for calibration
        :param bool merge_multiple_classes:
            Default True. If False, the given list of calibration-classes
            is handeled as-is. This means if you pass two CalibrationClass objects
            with the same name (e.g. "device on"), the calibration process will run
            for both these classes stand-alone.
            This will automatically yield an intersection of tuner-parameters, however may
            have advantages in some cases.
        :return:
            Returns a list of dictionaries. One dict is the SALib-result for
            one calibration-class. The order is based on the order of the
            calibration-class list.
        :rtype: list
        """
        # Check correct input
        calibration_classes = utils.validate_cal_class_input(calibration_classes)
        # Merge the classes for avoiding possible intersection of tuner-parameters
        if merge_multiple_classes:
            calibration_classes = data_types.merge_calibration_classes(calibration_classes)

        all_results = []
        for cal_class in calibration_classes:
            t_sen_start = time.time()
            self.logger.info('Start sensitivity analysis of class: %s, '
                             'Time-Interval: %s-%s s', cal_class.name,
                             cal_class.start_time, cal_class.stop_time)

            self.problem = self.create_problem(cal_class.tuner_paras)
            samples = self.generate_samples()
            # Generate list with metrics of every parameter variation
            output_array = self.simulate_samples(
                samples=samples,
                cal_class=cal_class)
            result = self.analysis_function(
                x=samples,
                y=output_array
            )
            t_sen_stop = time.time()
            result['duration[s]'] = t_sen_stop - t_sen_start
            all_results.append(result)
        result = self._conv_local_results(results=all_results,
                                          local_classes=calibration_classes)
        return result, calibration_classes

    @staticmethod
    def create_problem(tuner_paras) -> dict:
        """Create function for later access if multiple calibration-classes are used."""
        num_vars = len(tuner_paras.get_names())
        bounds = np.array(tuner_paras.get_bounds())
        problem = {'num_vars': num_vars,
                   'names': tuner_paras.get_names(),
                   'bounds': np.transpose(bounds)}
        return problem

    @staticmethod
    def select_by_threshold(calibration_classes, result, threshold):
        """
        Automatically select sensitive tuner parameters based on a given threshold
        and a key-word of the result.

        :param list calibration_classes:
            List of aixcalibuha.data_types.CalibrationClass objects that you want to
            automatically select sensitive tuner-parameters.
        :param pd.DataFrame result:
            Result object of sensitivity analysis run
        :param float threshold:
            Minimal required value of given key
        :return: list calibration_classes
        """
        for num_class, cal_class in enumerate(calibration_classes):
            class_result = result.loc[cal_class.name]
            tuner_paras = copy.deepcopy(cal_class.tuner_paras)
            select_names = class_result[class_result < threshold].index.values
            tuner_paras.remove_names(select_names)
            if not tuner_paras.get_names():
                raise ValueError(
                    'Automatic selection removed all tuner parameter '
                    f'from class {cal_class.name} after '
                    'SensitivityAnalysis was done. Please adjust the '
                    'threshold in json or manually chose tuner '
                    'parameters for the calibration.')
            # cal_class.set_tuner_paras(tuner_paras)
            cal_class.tuner_paras = tuner_paras
        return calibration_classes

    def _conv_global_result(self, result: dict, cal_class: CalibrationClass):
        glo_res_dict = self._get_res_dict(result=result, cal_class=cal_class)
        return pd.DataFrame(glo_res_dict, index=['global'])

    def _conv_local_results(self, results: list, local_classes: list):
        _conv_results = [self._get_res_dict(result=result, cal_class=local_class)
                         for result, local_class in zip(results, local_classes)]
        df = pd.DataFrame(_conv_results)
        df.index = [c.name for c in local_classes]
        return df

    @abc.abstractmethod
    def _get_res_dict(self, result: dict, cal_class: CalibrationClass):
        """
        Convert the result object to a dict with the key
        being the variable name and the value being the result
        associated to self.analysis_variable.
        """
        raise NotImplementedError
