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
import aixcalibuha
from aixcalibuha import CalibrationClass
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
        the simulation fails. See also: ret_val_on_error
    :keyword float,np.NAN ret_val_on_error
        Default is np.NAN. If fail_on_error is false, you can specify here
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
        Returns:
            List[str]: A list of strings
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
        :return np.array
            An array containing the evaluated differences for each sample
        """
        output = []
        initial_names = cal_class.tuner_paras.get_names()
        self.sim_api.set_sim_setup({"initialNames": initial_names,
                                           "startTime": cal_class.start_time,
                                           "stopTime": cal_class.stop_time})
        for i, initial_values in enumerate(samples):
            # Simulate the current values
            self.logger.info('Parameter variation %s of %s',
                             i+1, len(samples))
            self.sim_api.set_initial_values(initial_values)

            # Simulate
            # pylint: disable=broad-except
            try:
                # Generate the folder name for the calibration
                if self.save_files:
                    savepath_files = os.path.join(self.cd,
                                                  f"simulation_{i + 1}")
                    filepath = self.sim_api.simulate(savepath_files=savepath_files,
                                                     inputs=cal_class.inputs)
                    # Load the result file to the goals object
                    sim_target_data = data_types.TimeSeriesData(filepath)
                else:
                    target_sim_names = cal_class.goals.get_sim_var_names()
                    self.sim_api.set_sim_setup({"resultNames": target_sim_names})
                    df = self.sim_api.simulate(inputs=cal_class.inputs)
                    # Convert it to time series data object
                    sim_target_data = data_types.TimeSeriesData(df)
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
            calibration_classes = aixcalibuha.merge_calibration_classes(calibration_classes)

        all_results = []
        for cal_class in calibration_classes:
            t_sen_start = time.time()
            self.logger.info('Start sensitivity analysis of class: %s, '
                             'Time-Interval: %s-%s s', cal_class.name,
                             cal_class.start_time,cal_class.stop_time)

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
        return all_results, calibration_classes

    def automatic_run(self, calibration_classes):
        """
        Automated run according to (currently unpublished) method
        of Thomas Storek.
        Idea and concept will follow in future versions.
        Core is to first run a global SA and then a local SA.
        """
        # Check input
        calibration_classes = utils.validate_cal_class_input(calibration_classes)
        # Create one global class and run it
        global_classes = copy.deepcopy(calibration_classes)
        # Set the name to global
        for _class in global_classes:
            _class.name = "global"
        self.logger.info("Running global sensitivity analysis")
        global_res, global_classes = self.run(global_classes, merge_multiple_classes=True)
        global_class = global_classes[0]  # convert to scalar class
        # Run the local analysis
        self.logger.info("Running local sensitivity analysis")
        local_res, local_classes = self.run(calibration_classes, merge_multiple_classes=True)
        sorted_classes = self._automatic_ordering(global_res=global_res,
                                                  local_results=local_res,
                                                  unsorted_classes=local_classes,
                                                  global_class=global_class)
        self.logger.info("Finished automatic run.")
        return sorted_classes

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
    def select_by_threshold(calibration_classes, result, threshold, key="mu_star"):
        """
        Automatically select sensitive tuner parameters based on a given threshold
        and a key-word of the result.

        :param list calibration_classes:
            List of aixcalibuha.data_types.CalibrationClass objects that you want to
            automatically select sensitive tuner-parameters.
        :param list result:
            List of dicts (Sensitivity results)
        :param float threshold:
            Minimal required value of given key
        :param str key: Value that is used to define the sensitivity.
            Default is mu_star, "the absolute mean elementary effect"
            Choose between: mu, mu_star, sigma, mu_star_conf
        :return: list calibration_classes
        """
        for num_class, cal_class in enumerate(calibration_classes):
            class_result = result[num_class]
            tuner_paras = copy.deepcopy(cal_class.tuner_paras)
            select_names = []
            for i, sen_value in enumerate(class_result[key]):
                if sen_value < threshold:
                    select_names.append(class_result["names"][i])
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

    def _automatic_ordering(self, global_res, local_results,
                            unsorted_classes, global_class) -> list:
        self.logger.info("Automatically ordering calibration "
                         "classes based on global and local result")
        # Convert results to a dict containing only var_name: value
        glo_df = self._conv_global_result(result=global_res[0],
                                          cal_class=global_class)
        loc_df = self._conv_local_results(results=local_results,
                                          local_classes=unsorted_classes)
        self.logger.info("Starting recursive ordering of calibration classes.\n"
                         "Global results:\n %s\n"
                         "Local results:\n %s", glo_df, loc_df)
        sorted_names = self._recursive_cal_class_sorting(global_df=glo_df,
                                                         local_df=loc_df)
        # Sort class by name
        self.logger.info("Sort and return classes based on sorted names")
        sorted_classes = []
        for class_name in sorted_names:
            for local_class in unsorted_classes:
                if local_class.name == class_name:
                    sorted_classes.append(local_class)
        # Return sorted output
        return sorted_classes

    def _recursive_cal_class_sorting(self, global_df, local_df, sorted_list=None) -> list:
        """
        # 3. Vergleich: Finde TP_global[0] == TP_lokal[0]
        # if: nur in einer Klasse -> break
        # if: in mehreren Klassen dominant -> dann wo es ggü dem
        zweiten am besten ist (max(TP_lokal[0]/TP_lokal[1]))
        # Rekursiv: if: in keiner Klasse -> TP_global[0] == TP_lokal[1]
        # -> 1. zu kalibrierende Klasse mit TP_global[i]
        """
        if sorted_list is None:
            sorted_list = []
        # check if recursion is finished
        if local_df.empty or global_df.empty:
            # Remaining classes have no order
            sorted_list.extend(local_df.index.to_list())
            self.logger.info("Stopping recursion. Ordered classes are: %s",
                             ', '.join(sorted_list))
            return sorted_list

        # Get current max_var name and valu
        var_name = global_df.idxmax(axis=1).values[0]
        loc_max = local_df.idxmax(axis=1).copy()
        is_max = loc_max[loc_max == var_name]
        if len(is_max) <= 1:
            # Get maximal value
            class_name = local_df[var_name].idxmax()
        else:
            # Divide by max to get the maximal difference between first and second value
            local_max_df = local_df.loc[is_max.index]
            q_to_max = local_max_df.div(local_max_df[var_name],
                                        axis=0).copy().drop('heatConv_a', axis=1)
            q_to_max = q_to_max[q_to_max == q_to_max.max(axis=1).min()].dropna(how='all')
            class_name = q_to_max.index.values[0]
        sorted_list.append(class_name)
        local_df = local_df.drop(class_name)
        global_df = global_df.drop(var_name, axis=1)
        self.logger.info("Recursively selected class '%s' as class '%s'",
                         class_name, len(sorted_list))
        return self._recursive_cal_class_sorting(
            global_df=global_df,
            local_df=local_df,
            sorted_list=sorted_list
        )

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
        """Convert the result object to a dict with the key
        being the variable name and the value being the result
        associated to self.analysis_variable."""
        raise NotImplementedError
