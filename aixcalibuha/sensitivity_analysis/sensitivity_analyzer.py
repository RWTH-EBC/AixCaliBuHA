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
import matplotlib.pyplot as plt
from SALib.plotting.bar import plot as barplot


class SenAnalyzer(abc.ABC):
    """
    Class to perform a Sensitivity Analysis.

    :param SimulationAPI sim_api:
        Simulation-API used to simulate the samples
    :param int num_samples:
        The parameter `N` to the sampler methods of sobol and morris. NOTE: This is not the
        number of samples produced, but relates to the total number of samples produced in
        a manner dependent on the sampler method used. See the documentation of the specific 
        method in the SALib for more information.
    :keyword str or [str] analysis_variable: 
        Default is a list of all possible result values.
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
                                            self.analysis_variables)

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

    @property
    def analysis_variable(self):
        return self._analysis_variable

    @analysis_variable.setter
    def analysis_variable(self, value):
        if not isinstance(value, (list, tuple)):
            value = [value]
        false_values = []
        for v in value:
            if v not in self.analysis_variables:
                false_values.append(v)
        if false_values:
            raise ValueError(f'Given analysis_variable "{false_values}" not '
                             f'supported for class {self.__class__.__name__}. '
                             f'Supported options are: {", ".join(self.analysis_variables)}.')
        self._analysis_variable = value

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

    def simulate_samples(self, samples, cal_class, verbose=False, scale=False):
        """
        Put the parameters in the model and simulate it.

        :param Union[list, np.ndarray] samples:
            Output variables in dymola
        :param cal_class:
            One class for calibration. Goals and tuner_paras have to be set
        :param verbose:
            Default is False.
            If True returns an additional dict containing the evaluated differences
            for each sample as values in a np.array of the combined goal with the key 'all'
            and for the single goals with their VARIABLE_NAME as the key.
        :param scale:
            Default is False. If True the bounds of the tuner-parameters will be scaled between 0 and 1.

        :returns: np.array
            An array containing the evaluated differences for each sample
        """
        output = []
        list_output_verbose = []
        # Set the output interval according the given Goals
        mean_freq = cal_class.goals.get_meas_frequency()
        self.logger.info("Setting output_interval of simulation according "
                         "to measurement target data frequency: %s", mean_freq)
        self.sim_api.sim_setup.output_interval = mean_freq
        initial_names = cal_class.tuner_paras.get_names()
        self.sim_api.set_sim_setup({"start_time": cal_class.start_time,
                                    "stop_time": cal_class.stop_time})
        self.sim_api.result_names = cal_class.goals.get_sim_var_names()
        self.logger.info('Starting %s parameter variations on %s cores',
                         len(samples), self.sim_api.n_cpu)
        # Simulate the current values
        parameters = []
        for i, initial_values in enumerate(samples):
            if scale:
                initial_values = cal_class.tuner_paras.descale(initial_values)
            parameters.append({name: value for name, value in zip(initial_names, initial_values)})

        if self.save_files:
            result_file_names = [f"simulation_{idx + 1}" for idx in range(len(parameters))]
            _filepaths = self.sim_api.simulate(
                parameters=parameters,
                return_option="savepath",
                savepath=self.sim_api.cd,
                result_file_name=result_file_names,
                fail_on_error=self.fail_on_error,
                inputs=cal_class.inputs,
                **cal_class.input_kwargs
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
                parameters=parameters,
                inputs=cal_class.inputs,
                fail_on_error=self.fail_on_error,
                **cal_class.input_kwargs
            )
        self.logger.info('Finished %s simulations', len(samples))
        for i, result in enumerate(results):
            if result is None:
                output.append(self.ret_val_on_error)
            else:
                cal_class.goals.set_sim_target_data(result)
                cal_class.goals.set_relevant_time_intervals(cal_class.relevant_intervals)
                # Evaluate the current objective
                if verbose:
                    total_res, verbose_calculation = cal_class.goals.eval_difference(verbose=verbose)
                    output.append(total_res)
                    list_output_verbose.append(verbose_calculation)
                else:
                    total_res = cal_class.goals.eval_difference(verbose=verbose)
                    output.append(total_res)
        if verbose:
            # restructure output_verbose
            output_verbose = {}
            for key, val in list_output_verbose[0].items():
                output_verbose[key] = np.array([])
            for i in list_output_verbose:
                for key, val in i.items():
                    output_verbose[key] = np.append(output_verbose[key], np.array([val[1]]))
            return np.asarray(output), output_verbose
        return np.asarray(output)

    def run(self, calibration_classes, merge_multiple_classes=True, **kwargs):
        """
        Execute the sensitivity analysis for each class and
        return the result.

        :param CalibrationClass,list calibration_classes:
            Either one or multiple classes for calibration
        :param bool merge_multiple_classes:
            Default True. If False, the given list of calibration-classes
            is handled as-is. This means if you pass two CalibrationClass objects
            with the same name (e.g. "device on"), the calibration process will run
            for both these classes stand-alone.
            This will automatically yield an intersection of tuner-parameters, however may
            have advantages in some cases.
        :keyword bool verbose:
            Default False. If True, all sensitivity measures of the SALib function are calculated
            and returned. In addition to the combined Goals of the Classes (saved under index Goal: all),
            the sensitivity measures of the individual Goals will also be calculated and returned.
        :keyword bool plot_resutl:
            Default True. If True, the results will be plotted.
        :return:
            Returns a pandas.DataFrame. The DataFrame has a Multiindex with the
            levels Class, Goal and Analysis variable. The Goal name of combined goals is 'all'.
            The variables are the tuner-parameters.
            For the Sobol Method and calc_second_order returns a tuple of DataFrames (df_1, df_2)
            where df_2 contains the second oder analysis variables and has an extra index level
            Interaction, which also contians the variables.
        :rtype: pandas.DataFrame
        """
        verbose = kwargs.pop('verbose', False)
        scale = kwargs.pop('scale', False)
        plot_result = kwargs.pop('plot_result', True)
        # Check correct input
        calibration_classes = utils.validate_cal_class_input(calibration_classes)
        # Merge the classes for avoiding possible intersection of tuner-parameters
        if merge_multiple_classes:
            calibration_classes = data_types.merge_calibration_classes(calibration_classes)

        all_results = []
        for col, cal_class in enumerate(calibration_classes):
            t_sen_start = time.time()
            self.logger.info('Start sensitivity analysis of class: %s, '
                             'Time-Interval: %s-%s s', cal_class.name,
                             cal_class.start_time, cal_class.stop_time)

            self.problem = self.create_problem(cal_class.tuner_paras, scale=scale)
            samples = self.generate_samples()
            # Generate list with metrics of every parameter variation
            results_goals = {}
            output_array, output_verbose = self.simulate_samples(
                samples=samples,
                cal_class=cal_class,
                verbose=True)
            result = self.analysis_function(
                x=samples,
                y=output_array
            )
            t_sen_stop = time.time()
            result['duration[s]'] = t_sen_stop - t_sen_start
            results_goals['all'] = result
            if verbose:
                for key, val in output_verbose.items():
                    result_goal = self.analysis_function(
                        x=samples,
                        y=output_verbose[key]
                    )
                    results_goals[key] = result_goal
            all_results.append(results_goals)
        result = self._conv_local_results(results=all_results,
                                          local_classes=calibration_classes,
                                          verbose=verbose)
        if plot_result:
            self.plot(result)
        return result, calibration_classes

    @staticmethod
    def create_problem(tuner_paras, scale=False) -> dict:
        """Create function for later access if multiple calibration-classes are used."""
        num_vars = len(tuner_paras.get_names())
        bounds = np.array(tuner_paras.get_bounds())
        if scale:
            bounds = [np.zeros_like(bounds[0]), np.ones_like(bounds[1])]
        problem = {'num_vars': num_vars,
                   'names': tuner_paras.get_names(),
                   'bounds': np.transpose(bounds)}
        return problem

    @staticmethod
    def select_by_threshold(calibration_classes, result, threshold):
        # TODO: chang to fit the new result dataframes
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

    def _conv_local_results(self, results: list, local_classes: list, verbose=False):
        _conv_results = []
        tuples = []
        for class_results, local_class in zip(results, local_classes):
            for goal, goal_results in class_results.items():
                for av in self.analysis_variable:
                    _conv_results.append(self._get_res_dict(result=goal_results,
                                                            cal_class=local_class,
                                                            analysis_variable=av))
                    tuples.append((local_class.name, goal, av))
        index = pd.MultiIndex.from_tuples(tuples=tuples, names=['Class', 'Goal', 'Analysis variable'])
        df = pd.DataFrame(_conv_results, index=index)
        return df

    @abc.abstractmethod
    def _get_res_dict(self, result: dict, cal_class: CalibrationClass, analysis_variable: str):
        """
        Convert the result object to a dict with the key
        being the variable name and the value being the result
        associated to analysis_variable.
        """
        raise NotImplementedError

    def _del_duplicates(self, x):
        return list(dict.fromkeys(x))

    def _get_suffix(self, modelica_var_name):
        index_last_dot = modelica_var_name.rfind('.')
        suffix = modelica_var_name[index_last_dot + 1:]
        return suffix

    def plot_single(self, result: pd.DataFrame, **kwargs):
        '''
        Plot senitivity results of first and total order analysis variables.
        For each calibration class one figure is created, which shows for each goal an axis
        with a barplot of the values of the analysis variables.

        :param pd.DataFrame result:
            A result from run
        :keyword bool show_plot:
            Default is True. If False, all created plots are not shown.
        :keyword bool use_suffix:
            Default is True: If True, the last part after the last point
            of Modelica variables is used for the x ticks.
        :return:
            Returns all created figures and axes in lists like [fig], [ax]
        '''
        show_plot = kwargs.pop('show_plot', True)
        # kwargs for the design
        use_suffix = kwargs.pop('use_suffix', True)

        # get lists of the calibration classes their goals and the analysis variables in the result dataframe
        cal_classes = self._del_duplicates(list(result.index.get_level_values(0)))
        goals = self._del_duplicates(list(result.index.get_level_values(1)))
        analysis_variables = self._del_duplicates(list(result.index.get_level_values(2)))

        # rename tuner_names in result to the suffix of their variable name
        if use_suffix:
            tuner_names = list(result.columns)
            rename_tuner_names = {name: self._get_suffix(name) for name in tuner_names}
            result = result.rename(columns=rename_tuner_names)

        # when the index is not sorted pandas throws a performance warning
        result = result.sort_index()

        # plotting with simple plot function of the SALib
        figs = []
        axes = []
        for col, cal_class in enumerate(cal_classes):
            fig, ax = plt.subplots(len(goals), sharex='all')
            fig.suptitle(cal_class)
            figs.append(fig)
            if not isinstance(ax, np.ndarray):
                ax = [ax]
            axes.append(ax)
            for row, goal in enumerate(goals):
                result_df = result.loc[cal_class, goal]
                axes[col][row].grid(True, which='both', axis='y')
                barplot(result_df.T, ax=axes[col][row])
                axes[col][row].set_title(goal)
                axes[col][row].legend()

        if show_plot:
            plt.show()

        return figs, axes

    def plot(self, result):
        """
        Plot the results of the sensitivity analysis method from run().

        :param pd.DataFrame result:
            Dataframe of the results like from the run() function.
        :return tuple of matplotlib objects (fig, ax)
        """
        self.plot_single(result=result)

    @staticmethod
    def load_from_csv(path):
        result = pd.read_csv(path, index_col=[0, 1, 2])
        return result
