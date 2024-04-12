"""Package containing modules for sensitivity analysis.
The module contains the relevant base-classes."""
import abc
import copy
import os
import pathlib
import multiprocessing as mp
from typing import List
from collections import Counter
import numpy as np
import pandas as pd
from ebcpy.utils import setup_logger
from ebcpy.utils.reproduction import CopyFile
from ebcpy.simulationapi import SimulationAPI
from aixcalibuha import CalibrationClass, data_types
from aixcalibuha import utils
from aixcalibuha.sensitivity_analysis.plotting import plot_single, plot_time_dependent


def _load_single_file(_filepath, parquet_engine='pyarrow'):
    """Helper function"""
    if _filepath is None:
        return None
    return data_types.TimeSeriesData(_filepath, default_tag='sim', key='simulation',
                                     engine=parquet_engine)


def _load_files(_filepaths, parquet_engine='pyarrow'):
    """Helper function"""
    results = []
    for _filepath in _filepaths:
        results.append(_load_single_file(_filepath, parquet_engine=parquet_engine))
    return results


def _restruct_verbose(list_output_verbose):
    """Helper function"""
    output_verbose = {}
    for key, val in list_output_verbose[0].items():
        output_verbose[key] = np.array([])
    for i in list_output_verbose:
        for key, val in i.items():
            output_verbose[key] = np.append(output_verbose[key], np.array([val[1]]))
    return output_verbose


def _concat_all_sims(sim_results_list):
    """Helper function that concat all results in a list to one DataFrame."""
    sim_results_list = [r.to_df() for r in sim_results_list]
    sim_results_list = pd.concat(sim_results_list, keys=range(len(sim_results_list)),
                                 axis='columns')
    sim_results_list = sim_results_list.swaplevel(axis=1).sort_index(axis=1)
    return sim_results_list


def _restruct_time_dependent(sen_time_dependent_list, time_index):
    """Helper function that restructures the time dependent sensitivity results."""

    def _restruct_single(sen_time_dependent_list_s, second_order=False):
        sen_time_dependent_df = pd.concat(sen_time_dependent_list_s, keys=time_index, axis=0)
        sen_time_dependent_df = sen_time_dependent_df.droplevel('Class', axis='index')
        sen_time_dependent_df = sen_time_dependent_df.swaplevel(0, 1)
        sen_time_dependent_df = sen_time_dependent_df.swaplevel(1, 2).sort_index(axis=0)
        if second_order:
            sen_time_dependent_df = sen_time_dependent_df.swaplevel(2, 3).sort_index(axis=0)
            sen_time_dependent_df.index.set_names(
                ['Goal', 'Analysis variable', 'Interaction', 'time'], inplace=True)
        else:
            sen_time_dependent_df.index.set_names(['Goal', 'Analysis variable', 'time'],
                                                  inplace=True)
        return sen_time_dependent_df

    if isinstance(sen_time_dependent_list[0], tuple):
        sen_time_dependent_list1, sen_time_dependent_list2 = zip(*sen_time_dependent_list)
        return _restruct_single(sen_time_dependent_list1), _restruct_single(
            sen_time_dependent_list2, True)
    return _restruct_single(sen_time_dependent_list)


def _divide_chunks(long_list, chunk_length):
    """Helper function that divides all list into multiple list with a specific chunk length."""
    for i in range(0, len(long_list), chunk_length):
        yield long_list[i:i + chunk_length]


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
    :keyword str,os.path.normpath cd:
        The path for the current working directory.
        Logger and results will be stored here.
    :keyword boolean fail_on_error:
        Default is False. If True, the calibration will stop with an error if
        the simulation fails. See also: ``ret_val_on_error``
    :keyword float,np.NAN ret_val_on_error:
        Default is np.NAN. If ``fail_on_error`` is false, you can specify here
        which value to return in the case of a failed simulation. Possible
        options are np.NaN, np.inf or some other high numbers. be aware that this
        max influence the solver.
    :keyword boolean save_files:
        Default False. If true, all simulation files for each iteration will be saved!
    :keyword str suffix_files:
        Default 'csv'. Specifies the data format to store the simulation files in.
        Options are 'csv', 'hdf', 'parquet'.
    :keyword str parquet_engine:
        The engine to use for the data format parquet.
        Supported options can be extracted
        from the ebcpy.TimeSeriesData.save() function.
        Default is 'pyarrow'.
    :keyword str,os.path.normpath savepath_sim:
        Default is cd. Own directory for the time series data sets of all simulations
        during the sensitivity analysis. The own dir can be necessary for large data sets,
        because they can crash IDE during indexing when they are in the project folder.

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
        self.suffix_files = kwargs.pop('suffix_files', 'csv')
        self.parquet_engine = kwargs.pop('parquet_engine', 'pyarrow')
        self.ret_val_on_error = kwargs.pop("ret_val_on_error", np.NAN)
        self.cd = kwargs.pop("cd", os.getcwd())
        self.savepath_sim = kwargs.pop('savepath_sim', self.cd)

        if isinstance(self.cd, str):
            self.cd = pathlib.Path(self.cd)
        if isinstance(self.savepath_sim, str):
            self.savepath_sim = pathlib.Path(self.savepath_sim)

        # Setup the logger
        self.logger = setup_logger(working_directory=self.cd, name=self.__class__.__name__)

        # Setup default values
        self.problem: dict = None
        self.reproduction_files = []

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

    def simulate_samples(self, cal_class, **kwargs):
        """
        Creates the samples for the calibration class and simulates them.

        :param cal_class:
            One class for calibration. Goals and tuner_paras have to be set
        :keyword scale:
            Default is False. If True the bounds of the tuner-parameters
            will be scaled between 0 and 1.

        :return:
            Returns two lists. First a list with the simulation results for each sample.
            If save_files the list contains the filepaths to the results
            Second a list of the samples.
        :rtype: list
        """
        scale = kwargs.pop('scale', False)
        # Set the output interval according the given Goals
        mean_freq = cal_class.goals.get_meas_frequency()
        self.logger.info("Setting output_interval of simulation according "
                         "to measurement target data frequency: %s", mean_freq)
        self.sim_api.sim_setup.output_interval = mean_freq
        initial_names = cal_class.tuner_paras.get_names()
        self.sim_api.set_sim_setup({"start_time": cal_class.start_time,
                                    "stop_time": cal_class.stop_time})
        self.sim_api.result_names = cal_class.goals.get_sim_var_names()

        self.problem = self.create_problem(cal_class.tuner_paras, scale=scale)
        samples = self.generate_samples()

        # creat df of samples with the result_file_names as the index
        result_file_names = [f"simulation_{idx}" for idx in range(len(samples))]
        samples_df = pd.DataFrame(samples, columns=initial_names, index=result_file_names)
        samples_df.to_csv(self.cd.joinpath(f'samples_{cal_class.name}.csv'))

        # Simulate the current values
        parameters = []
        for initial_values in samples:
            if scale:
                initial_values = cal_class.tuner_paras.descale(initial_values)
            parameters.append(dict(zip(initial_names, initial_values)))

        self.logger.info('Starting %s parameter variations on %s cores',
                         len(samples), self.sim_api.n_cpu)
        if self.save_files:
            sim_dir = self.savepath_sim.joinpath(f'simulations_{cal_class.name}')
            os.makedirs(sim_dir, exist_ok=True)
            samples_df.to_csv(self.savepath_sim.joinpath(f'samples_{cal_class.name}.csv'))
            self.logger.info(f'Saving simulation files in: {sim_dir}')
            _filepaths = self.sim_api.simulate(
                parameters=parameters,
                return_option="savepath",
                savepath=sim_dir,
                result_file_name=result_file_names,
                result_file_suffix=self.suffix_files,
                parquet_engine=self.parquet_engine,
                fail_on_error=self.fail_on_error,
                inputs=cal_class.inputs,
                **cal_class.input_kwargs
            )
            self.reproduction_files.extend(_filepaths)
            results = _filepaths
        else:
            results = self.sim_api.simulate(
                parameters=parameters,
                inputs=cal_class.inputs,
                fail_on_error=self.fail_on_error,
                **cal_class.input_kwargs
            )
        self.logger.info('Finished %s simulations', len(samples))
        return results, samples

    def _check_index(self, tsd: data_types.TimeSeriesData, sim_num=None):
        freq = tsd.frequency
        if sim_num is None:
            sim_num = tsd.filepath.name
        if freq[0] != self.sim_api.sim_setup.output_interval:
            self.logger.info(
                f'The mean value of the frequency from {sim_num} does not match output '
                'interval index will be cleaned and spaced equally')
            tsd.to_datetime_index()
            tsd.clean_and_space_equally(f'{str(self.sim_api.sim_setup.output_interval * 1000)}ms')
            tsd.to_float_index()
            freq = tsd.frequency
        if freq[1] > 0.0:
            self.logger.info(f'The standard deviation of the frequency from {sim_num} is to high '
                             f'and will be rounded to the accuracy of the output interval')
            tsd.index = np.round(tsd.index.astype("float64"),
                                 str(self.sim_api.sim_setup.output_interval)[::-1].find('.'))
        return tsd

    def _single_eval_statistical_measure(self, kwargs_eval):
        """Evaluates statistical measure of one result"""
        cal_class = kwargs_eval.pop('cal_class')
        result = kwargs_eval.pop('result')
        num_sim = kwargs_eval.pop('sim_num', None)
        if result is None:
            verbose_error = {}
            for goal, weight in zip(cal_class.goals.get_goals_list(), cal_class.goals.weightings):
                verbose_error[goal] = (weight, self.ret_val_on_error)
            return self.ret_val_on_error, verbose_error
        result = self._check_index(result, num_sim)
        cal_class.goals.set_sim_target_data(result)
        cal_class.goals.set_relevant_time_intervals(cal_class.relevant_intervals)
        # Evaluate the current objective
        total_res, verbose_calculation = cal_class.goals.eval_difference(verbose=True)
        return total_res, verbose_calculation

    def eval_statistical_measure(self, cal_class, results, verbose=True):
        """Evaluates statistical measures of results on single core"""
        self.logger.info('Starting evaluation of statistical measure')
        output = []
        list_output_verbose = []
        for i, result in enumerate(results):
            total_res, verbose_calculation = self._single_eval_statistical_measure(
                {'cal_class': cal_class, 'result': result, 'sim_num': f'simulation_{i}'}
            )
            output.append(total_res)
            list_output_verbose.append(verbose_calculation)
        if verbose:
            # restructure output_verbose
            output_verbose = _restruct_verbose(list_output_verbose)
            return np.asarray(output), output_verbose
        return np.asarray(output)

    def _single_load_eval_file(self, kwargs_load_eval):
        """For multiprocessing"""
        filepath = kwargs_load_eval.pop('filepath')
        _result = _load_single_file(filepath, self.parquet_engine)
        kwargs_load_eval.update({'result': _result})
        total_res, verbose_calculation = self._single_eval_statistical_measure(kwargs_load_eval)
        return total_res, verbose_calculation

    def _mp_load_eval(self, _filepaths, cal_class, n_cpu):
        """
        Loading and evaluating the statistical measure of saved simulation files on multiple cores
        """
        self.logger.info(f'Load files and evaluate statistical measure on {n_cpu} processes.')
        kwargs_load_eval = []
        for filepath in _filepaths:
            kwargs_load_eval.append({'filepath': filepath, 'cal_class': cal_class})
        output_array = []
        list_output_verbose = []
        with mp.Pool(processes=n_cpu) as pool:
            for total, verbose in pool.imap(self._single_load_eval_file, kwargs_load_eval):
                output_array.append(total)
                list_output_verbose.append(verbose)
            output_array = np.asarray(output_array)
            output_verbose = _restruct_verbose(list_output_verbose)
        return output_array, output_verbose

    def _load_eval(self, _filepaths, cal_class, n_cpu):
        """
        Loading and evaluating the statistical measure of saved simulation files.
        Single- or multiprocessing possible with definition of n_cpu.
        """
        if n_cpu == 1:
            results = _load_files(_filepaths, self.parquet_engine)
            output_array, output_verbose = self.eval_statistical_measure(
                cal_class=cal_class,
                results=results
            )
            return output_array, output_verbose
        output_array, output_verbose = self._mp_load_eval(_filepaths, cal_class, n_cpu)
        return output_array, output_verbose

    def run(self, calibration_classes, merge_multiple_classes=True, **kwargs):
        """
        Execute the sensitivity analysis for each class and
        return the result.

        :param CalibrationClass,list calibration_classes:
            Either one or multiple classes for calibration with same tuner-parameters.
        :param bool merge_multiple_classes:
            Default True. If False, the given list of calibration-classes
            is handled as-is. This means if you pass two CalibrationClass objects
            with the same name (e.g. "device on"), the calibration process will run
            for both these classes stand-alone.
            This will automatically yield an intersection of tuner-parameters, however may
            have advantages in some cases.
        :keyword bool verbose:
            Default False. If True, in addition to the combined Goals of the Classes
            (saved under index Goal: all), the sensitivity measures of the individual
            Goals will also be calculated and returned.
        :keyword scale:
            Default is False. If True the bounds of the tuner-parameters
            will be scaled between 0 and 1.
        :keyword bool use_fist_sim:
            Default False. If True, the simulations of the first calibration class will be used for
            all other calibration classes with their relevant time intervals.
            The simulations must be stored on a hard-drive, so it must be used with
            either save_files or load_files.
        :keyword int n_cpu:
            Default is 1. The number of processes to use for the evaluation of the statistical
            measure. For n_cpu > 1 only one simulation file is loaded at once in a process and
            dumped directly after the evaluation of the statistical measure,
            so that only minimal memory is used.
            Use this option for large analyses.
            Only implemented for save_files=True or load_sim_files=True.
        :keyword bool load_sim_files:
            Default False. If True, no new simulations are done and old simulations are loaded.
            The simulations and corresponding samples will be loaded from self.savepath_sim like
            they were saved from self.save_files. Currently, the name of the sim folder must be
            "simulations_CAL_CLASS_NAME" and for the samples "samples_CAL_CLASS_NAME".
            The usage of the same simulations for different
            calibration classes is not supported yet.
        :keyword bool save_results:
            Default True. If True, all results are saved as a csv in cd.
            (samples, statistical measures and analysis variables).
        :keyword bool plot_result:
            Default True. If True, the results will be plotted.
        :return:
            Returns a pandas.DataFrame. The DataFrame has a Multiindex with the
            levels Class, Goal and Analysis variable. The Goal name of combined goals is 'all'.
            The variables are the tuner-parameters.
            For the Sobol Method and calc_second_order returns a tuple of DataFrames (df_1, df_2)
            where df_2 contains the second oder analysis variables and has an extra index level
            Interaction, which also contains the variables.
        :rtype: pandas.DataFrame
        """
        verbose = kwargs.pop('verbose', False)
        scale = kwargs.pop('scale', False)
        use_first_sim = kwargs.pop('use_first_sim', False)
        n_cpu = kwargs.pop('n_cpu', 1)
        save_results = kwargs.pop('save_results', True)
        plot_result = kwargs.pop('plot_result', True)
        load_sim_files = kwargs.pop('load_sim_files', False)
        # Check correct input
        calibration_classes = utils.validate_cal_class_input(calibration_classes)
        # Merge the classes for avoiding possible intersection of tuner-parameters
        if merge_multiple_classes:
            calibration_classes = data_types.merge_calibration_classes(calibration_classes)

        # Check n_cpu
        if n_cpu > mp.cpu_count():
            raise ValueError(f"Given n_cpu '{n_cpu}' is greater "
                             "than the available number of "
                             f"cpus on your machine '{mp.cpu_count()}'")

        # Check if the usage of the simulations from the first calibration class for all is possible
        if use_first_sim:
            if not self.save_files and not load_sim_files:
                raise AttributeError('To use the simulations of the first calibration class '
                                     'for all classes the simulation files must be saved. '
                                     'Either set save_files=True or load already exiting files '
                                     'with load_sim_files=True.')
            start_time = 0
            stop_time = 0
            for idx, cal_class in enumerate(calibration_classes):
                if idx == 0:
                    start_time = cal_class.start_time
                    stop_time = cal_class.stop_time
                    continue
                if start_time > cal_class.start_time or stop_time < cal_class.stop_time:
                    raise ValueError(f'To use the simulations of the first calibration class '
                                     f'for all classes the start and stop times of the other '
                                     f'classes must be in the interval [{start_time}, {stop_time}] '
                                     f'of the first calibration class.')

        all_results = []
        for idx, cal_class in enumerate(calibration_classes):

            self.logger.info('Start sensitivity analysis of class: %s, '
                             'Time-Interval: %s-%s s', cal_class.name,
                             cal_class.start_time, cal_class.stop_time)

            # Generate list with metrics of every parameter variation
            results_goals = {}
            if load_sim_files:
                self.problem = self.create_problem(cal_class.tuner_paras, scale=scale)
                if use_first_sim:
                    class_name = calibration_classes[0].name
                else:
                    class_name = cal_class.name
                sim_dir = self.savepath_sim.joinpath(f'simulations_{class_name}')
                samples_path = self.savepath_sim.joinpath(f'samples_{class_name}.csv')
                self.logger.info(f'Loading samples from {samples_path}')
                samples = pd.read_csv(samples_path,
                                      header=0,
                                      index_col=0)
                samples = samples.to_numpy()
                result_file_names = [f"simulation_{idx}.{self.suffix_files}" for idx in
                                     range(len(samples))]
                _filepaths = [sim_dir.joinpath(result_file_name) for result_file_name in
                              result_file_names]
                self.logger.info(f'Loading simulation files from {sim_dir}')
                output_array, output_verbose = self._load_eval(_filepaths, cal_class, n_cpu)
            else:
                results, samples = self.simulate_samples(
                    cal_class=cal_class,
                    scale=scale
                )
                if self.save_files:
                    output_array, output_verbose = self._load_eval(results, cal_class, n_cpu)
                else:
                    output_array, output_verbose = self.eval_statistical_measure(
                        cal_class=cal_class,
                        results=results
                    )
                if use_first_sim:
                    load_sim_files = True

            # combine output_array and output_verbose
            # set key for output_array depending on one or multiple goals
            stat_mea = {'all': output_array}
            if len(output_verbose) == 1:
                stat_mea = output_verbose
            if len(output_verbose) > 1 and verbose:
                stat_mea.update(output_verbose)

            # save statistical measure and corresponding samples for each cal_class in cd
            if save_results:
                result_file_names = [f"simulation_{idx}" for idx in range(len(output_array))]
                stat_mea_df = pd.DataFrame(stat_mea, index=result_file_names)
                savepath_stat_mea = self.cd.joinpath(
                    f'{cal_class.goals.statistical_measure}_{cal_class.name}.csv')
                stat_mea_df.to_csv(savepath_stat_mea)
                self.reproduction_files.append(savepath_stat_mea)
                samples_df = pd.DataFrame(samples, columns=cal_class.tuner_paras.get_names(),
                                          index=result_file_names)
                savepath_samples = self.cd.joinpath(f'samples_{cal_class.name}.csv')
                samples_df.to_csv(savepath_samples)
                self.reproduction_files.append(savepath_samples)

            self.logger.info('Starting calculation of analysis variables')
            for key, val in stat_mea.items():
                result_goal = self.analysis_function(
                    x=samples,
                    y=val
                )
                results_goals[key] = result_goal
            all_results.append(results_goals)
            self.logger.info('Finished sensitivity analysis of class: %s, '
                             'Time-Interval: %s-%s s', cal_class.name,
                             cal_class.start_time, cal_class.stop_time)
        result = self._conv_local_results(results=all_results,
                                          local_classes=calibration_classes)
        if save_results:
            self._save(result)
        if plot_result:
            self.plot(result)
        return result, calibration_classes

    def _save(self, result: pd.DataFrame, time_dependent: bool = False):
        """
        Saves the result DataFrame of run and run_time_dependent.
        Needs to be overwritten for Sobol results.
        """
        if time_dependent:
            savepath_result = self.cd.joinpath(f'{self.__class__.__name__}_results_time.csv')
        else:
            savepath_result = self.cd.joinpath(f'{self.__class__.__name__}_results.csv')
        result.to_csv(savepath_result)
        self.reproduction_files.append(savepath_result)

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
    def select_by_threshold(calibration_classes, result, analysis_variable, threshold):
        """
        Automatically select sensitive tuner parameters based on a given threshold
        of a given analysis variable from a sensitivity result.
        Uses only the combined goals.

        :param list calibration_classes:
            List of aixcalibuha.data_types.CalibrationClass objects that you want to
            automatically select sensitive tuner-parameters.
        :param pd.DataFrame result:
            Result object of sensitivity analysis run
        :param str analysis_variable:
            Analysis variable to use for the selection
        :param float threshold:
            Minimal required value of given key
        :return: list calibration_classes
        """
        for cal_class in calibration_classes:
            first_goal = result.index.get_level_values(1)[0]
            class_result = result.loc[cal_class.name, first_goal, analysis_variable]
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

    @staticmethod
    def select_by_threshold_verbose(calibration_class: CalibrationClass,
                                    result: pd.DataFrame,
                                    analysis_variable: str,
                                    threshold: float,
                                    calc_names_for_selection: List[str] = None):
        """
        Select tuner-parameters of single calibration class with verbose sensitivity results.
        This function selects tuner-parameters if their sensitivity is equal or greater
        than the threshold in just one target value of one calibration class in the
        sensitivity result. This can be more robust because a small sensitivity in one target
        value and state of the system can mean that the parameter can also be calibrated in
        a global calibration class which calibrates multiple states and target values at
        the same time and has there not directly the same sensitivity as in the isolated
        view of a calibration class for only one state.

        :param CalibrationClass calibration_class:
            The calibration class from which the tuner parameters will be selected.
        :param pd.DataFrame result:
            Sensitivity results to use for the selection. Can include multiple classes.
        :param str analysis_variable:
            The analysis variable to use for the selection.
        :param float threshold:
            Minimal required value of given analysis variable.
        :param List[str] calc_names_for_selection:
            Specifies which calibration classes in the sensitivity results will be used for
            the selection. Default are all classes.
        """
        if Counter(calibration_class.tuner_paras.get_names()) != Counter(list(result.columns)):
            raise NameError("The tuner-parameter of the calibration class do not "
                            "match the tuner-parameters in the sensitivity result."
                            "They have to match.")

        result = result.loc[:, :, analysis_variable]
        calc_names_results = result.index.get_level_values("Class").unique()
        if calc_names_for_selection:
            for cal_class in calc_names_for_selection:
                if cal_class not in calc_names_results:
                    raise NameError(f"The calibration class name {cal_class} "
                                    f"does not match any class name "
                                    f"in the given sensitivity result.")
            result = result.loc[calc_names_for_selection, :, :]

        selected_tuners = (result >= threshold).any()

        remove_tuners = []
        for tuner, selected in selected_tuners.items():
            if not selected:
                remove_tuners.append(tuner)
        tuner_paras = copy.deepcopy(calibration_class.tuner_paras)
        tuner_paras.remove_names(remove_tuners)
        if not tuner_paras.get_names():
            raise ValueError("Threshold to small. All tuner-parameters would be removed.")
        calibration_class.tuner_paras = tuner_paras
        return calibration_class

    def run_time_dependent(self, cal_class: CalibrationClass, **kwargs):
        """
        Calculate the time dependent sensitivity for all the single goals in the calibration class.

        :param CalibrationClass cal_class:
            Calibration class with tuner-parameters to calculate sensitivity for.
            Can include dummy target date.
        :keyword scale:
            Default is False. If True the bounds of the tuner-parameters
            will be scaled between 0 and 1.
        :keyword bool load_sim_files:
            Default False. If True, no new simulations are done and old simulations are loaded.
            The simulations and corresponding samples will be loaded from self.savepath_sim like
            they were saved from self.save_files. Currently, the name of the sim folder must be
            "simulations_CAL_CLASS_NAME" and for the samples "samples_CAL_CLASS_NAME".
        :keyword bool save_results:
            Default True. If True, all results are saved as a csv in cd.
            (samples and analysis variables).
        :keyword int n_steps:
            Default is all time steps. If the problem is large, the evaluation of all time steps
            at once can cause a memory error. Then n_steps defines how many time_steps
            are evaluated at once in chunks. This increases the needed time exponentially and
            the simulation files must be saved.
        :keyword bool plot_result:
            Default True. If True, the results will be plotted.
        :return:
            Returns a pandas.DataFrame.
        :rtype: pandas.DataFrame
        """
        scale = kwargs.pop('scale', False)
        save_results = kwargs.pop('save_results', True)
        plot_result = kwargs.pop('plot_result', True)
        load_sim_files = kwargs.pop('load_sim_files', False)
        n_steps = kwargs.pop('n_steps', 'all')

        self.logger.info("Start time dependent sensitivity analysis.")
        if load_sim_files:
            self.problem = self.create_problem(cal_class.tuner_paras, scale=scale)
            sim_dir = self.savepath_sim.joinpath(f'simulations_{cal_class.name}')
            samples_path = self.savepath_sim.joinpath(f'samples_{cal_class.name}.csv')
            samples = pd.read_csv(samples_path,
                                  header=0,
                                  index_col=0)
            samples = samples.to_numpy()
            result_file_names = [f"simulation_{idx}.{self.suffix_files}" for idx in
                                 range(len(samples))]
            _filepaths = [sim_dir.joinpath(result_file_name) for result_file_name in
                          result_file_names]

            sen_time_dependent_list, time_index = self._load_analyze_tsteps(_filepaths=_filepaths,
                                                                            samples=samples,
                                                                            n_steps=n_steps,
                                                                            cal_class=cal_class)
            sen_time_dependent_df = _restruct_time_dependent(sen_time_dependent_list, time_index)
        else:
            results, samples = self.simulate_samples(
                cal_class=cal_class,
                scale=scale
            )
            if self.save_files:
                sen_time_dependent_list, time_index = self._load_analyze_tsteps(_filepaths=results,
                                                                                samples=samples,
                                                                                n_steps=n_steps,
                                                                                cal_class=cal_class)
                sen_time_dependent_df = _restruct_time_dependent(sen_time_dependent_list,
                                                                 time_index)
            else:
                variables = results[0].get_variable_names()
                time_index = results[0].index.to_numpy()
                total_result = _concat_all_sims(results)
                sen_time_dependent_list = []
                for time_step in time_index:
                    result_df_tstep = self._analyze_tstep_df(time_step=time_step,
                                                             tsteps_sim_results=total_result,
                                                             variables=variables,
                                                             samples=samples,
                                                             cal_class=cal_class)
                    sen_time_dependent_list.append(result_df_tstep)
                sen_time_dependent_df = _restruct_time_dependent(sen_time_dependent_list,
                                                                 time_index)
        self.logger.info("Finished time dependent sensitivity analysys.")
        if save_results:
            self._save(sen_time_dependent_df, time_dependent=True)
        if plot_result:
            if isinstance(sen_time_dependent_df, pd.DataFrame):
                plot_time_dependent(sen_time_dependent_df)
            else:
                plot_time_dependent(sen_time_dependent_df[0])
        return sen_time_dependent_df

    def _analyze_tstep_df(self, time_step, tsteps_sim_results, variables, samples, cal_class):
        """Analyze the sensitivity at a single time step."""
        result_dict_tstep = {}
        for var in variables:
            result_tstep_var = tsteps_sim_results[var].loc[time_step].to_numpy()
            if np.all(result_tstep_var == result_tstep_var[0]):
                sen_tstep_var = None
            else:
                sen_tstep_var = self.analysis_function(
                    x=samples,
                    y=result_tstep_var
                )
            result_dict_tstep[var] = sen_tstep_var
        result_df_tstep = self._conv_local_results(results=[result_dict_tstep],
                                                   local_classes=[cal_class])
        return result_df_tstep

    def _load_tsteps_df(self, tsteps, _filepaths):
        """
        Load all simulations and extract and concat the sim results of the time steps in tsteps.
        """
        self.logger.info(
            f"Loading time steps from {tsteps[0]} to {tsteps[-1]} of the simulation files.")
        tsteps_sim_results = []
        for _filepath in _filepaths:
            sim = _load_single_file(_filepath)
            tsteps_sim_results.append(sim.loc[tsteps[0]:tsteps[-1]])
        tsteps_sim_results = _concat_all_sims(tsteps_sim_results)
        return tsteps_sim_results

    def _load_analyze_tsteps(self, _filepaths, samples, n_steps, cal_class):
        """
        Load and analyze all time steps in chunks with n_steps time steps.
        """
        sim1 = _load_single_file(_filepaths[0])
        time_index = sim1.index.to_numpy()
        variables = sim1.get_variable_names()
        sen_time_dependent_list = []
        if n_steps == 'all':
            list_tsteps = [time_index]
        elif isinstance(n_steps, int) and not (n_steps <= 0 or n_steps > len(time_index)):
            list_tsteps = _divide_chunks(time_index, n_steps)
        else:
            raise ValueError(
                f"n_steps can only be between 1 and {len(time_index)} or the string all.")

        for tsteps in list_tsteps:
            tsteps_sim_results = self._load_tsteps_df(tsteps=tsteps, _filepaths=_filepaths)
            self.logger.info("Analyzing these time steps.")
            for tstep in tsteps:
                result_df_tstep = self._analyze_tstep_df(time_step=tstep,
                                                         tsteps_sim_results=tsteps_sim_results,
                                                         variables=variables,
                                                         samples=samples,
                                                         cal_class=cal_class)
                sen_time_dependent_list.append(result_df_tstep)
        return sen_time_dependent_list, time_index

    def _conv_global_result(self, result: dict, cal_class: CalibrationClass,
                            analysis_variable: str):
        glo_res_dict = self._get_res_dict(result=result, cal_class=cal_class,
                                          analysis_variable=analysis_variable)
        return pd.DataFrame(glo_res_dict, index=['global'])

    def _conv_local_results(self, results: list, local_classes: list):
        """
        Convert the result dictionaries form SALib of each class and goal into one DataFrame.
        Overwritten for Sobol.
        """
        _conv_results = []
        tuples = []
        for class_results, local_class in zip(results, local_classes):
            for goal, goal_results in class_results.items():
                for analysis_var in self.analysis_variables:
                    _conv_results.append(self._get_res_dict(result=goal_results,
                                                            cal_class=local_class,
                                                            analysis_variable=analysis_var))
                    tuples.append((local_class.name, goal, analysis_var))
        index = pd.MultiIndex.from_tuples(tuples=tuples,
                                          names=['Class', 'Goal', 'Analysis variable'])
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

    def plot(self, result):
        """
        Plot the results of the sensitivity analysis method from run().

        :param pd.DataFrame result:
            Dataframe of the results like from the run() function.
        :return tuple of matplotlib objects (fig, ax):
        """
        plot_single(result=result)

    @staticmethod
    def load_from_csv(path):
        """
        Load sensitivity results which were saved with the run() or run_time_dependent() function.

        For second order results use the load_second_order_from_csv() function of the SobolAnalyzer.
        """
        result = pd.read_csv(path, index_col=[0, 1, 2])
        return result

    def save_for_reproduction(self,
                              title: str,
                              path: pathlib.Path = None,
                              files: list = None,
                              exclude_sim_files: bool = False,
                              remove_saved_files: bool = False,
                              **kwargs):
        """
        Save the settings of the SenAnalyzer and SimApi in order to
        reproduce the simulations and sensitivity analysis method.
        All saved results will be also saved in the reproduction
        archive. The simulations can be excluded from saving.

        :param str title:
            Title of the study
        :param pathlib.Path path:
            Where to store the .zip file. If not given, self.cd is used.
        :param list files:
            List of files to save along the standard ones.
            Examples would be plots, tables etc.
        :param bool exclude_sim_files:
            Default False. If True, the simulation files will not be saved in
            the reproduction archive.
        :param bool remove_saved_files:
            Default False. If True, the result and simulation files will be moved
            instead of just copied.
        :param dict kwargs:
            All keyword arguments except title, files, and path of the function
            `save_reproduction_archive`. Most importantly, `log_message` may be
            specified to avoid input during execution.
        """
        if files is None:
            files = []

        for file_path in self.reproduction_files:
            if exclude_sim_files:
                if 'simulation' in str(file_path):
                    continue
            filename = "SenAnalyzer" + str(file_path).rsplit(self.cd.name, maxsplit=1)[-1]
            files.append(CopyFile(
                sourcepath=file_path,
                filename=filename,
                remove=remove_saved_files
            ))

        return self.sim_api.save_for_reproduction(
            title=title,
            path=path,
            files=files,
            **kwargs
        )
