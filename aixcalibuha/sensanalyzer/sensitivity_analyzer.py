"""Package containing modules for sensitivity analysis.
The module contains the relevant base-classes."""
import abc
import copy
import os
import numpy as np
from ebcpy.utils import setup_logger
from ebcpy import data_types
from ebcpy import simulationapi
import aixcalibuha
from aixcalibuha import utils


class SenAnalyzer(abc.ABC):
    """
    Class to perform a Sensitivity Analysis.

    :param simulationapi.SimulationAPI sim_api:
        Simulation-API used to simulate the samples
    :param SensitivityProblem sensitivity_problem:
        Parameter class for the sensitivity. it contains the demand of the sampler
        and create the dictionary parameter Problem
    :param int num_samples:
        The parameter `N` to the sampler methods of sobol and morris. NOTE: This is not the
        the number of samples produced, but relates to the total number of samples produced in
        a manner dependent on the sampler method used. See the documentation of sobol and
        morris in the SALib for more information.
    :param str statistical_measure:
        Used to evaluate the difference of simulated and measured data.
        Like "RMSE", "MAE" etc. See utils.statistics_analyzer.py for
        further info.
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

    def __init__(self, sim_api, sensitivity_problem, num_samples,
                 statistical_measure, **kwargs):
        """Instantiate class parameters"""
        # Setup the instance attributes
        self.sim_api = sim_api
        self.statistical_measure = statistical_measure
        self.sensitivity_problem = sensitivity_problem
        self.num_samples = num_samples

        # Update kwargs
        self.fail_on_error = kwargs.pop("fail_on_error", True)
        self.save_files = kwargs.pop("save_files", False)
        self.ret_val_on_error = kwargs.pop("ret_val_on_error", np.NAN)
        self.cd = kwargs.pop("cd", os.getcwd())

        # Setup the logger
        self.logger = setup_logger(cd=self.cd, name=self.__class__.__name__)

        # Setup default values
        self.problem: dict = None

    @property
    @abc.abstractmethod
    def analysis_variables(self):
        raise NotImplementedError(f'{self.__class__.__name__}.analysis_variables '
                                  f'property is not defined yet')

    @abc.abstractmethod
    def analysis_function(self, x, y):
        raise NotImplementedError(f'{self.__class__.__name__}.analysis_function '
                                  f'function is not defined yet')

    @abc.abstractmethod
    def create_sampler_demand(self):
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
        Put the parameter in dymola model, run it.

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
            self.logger.info(f'Parameter variation {i+1} of {len(samples)}')
            self.sim_api.set_initial_values(initial_values)

            # Simulate
            try:
                # Generate the folder name for the calibration
                if self.save_files:
                    savepath_files = os.path.join(self.cd,
                                                  f"simulation_{i + 1}")
                    filepath = self.sim_api.simulate(savepath_files=savepath_files)
                    # Load the result file to the goals object
                    sim_target_data = data_types.TimeSeriesData(filepath)
                else:
                    target_sim_names = cal_class.goals.get_sim_var_names()
                    self.sim_api.set_sim_setup({"resultNames": target_sim_names})
                    df = self.sim_api.simulate()
                    # Convert it to time series data object
                    sim_target_data = data_types.TimeSeriesData(df)
            except Exception as e:
                if self.fail_on_error:
                    raise e
                else:
                    return self.ret_val_on_error

            cal_class.goals.set_sim_target_data(sim_target_data)
            cal_class.goals.set_relevant_time_intervals(cal_class.relevant_intervals)

            # Evaluate the current objective
            total_res = cal_class.goals.eval_difference(self.statistical_measure)
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
            self.logger.info(f'Start sensitivity analysis of class: {cal_class.name}, '
                             f'Time-Interval: {cal_class.start_time}-{cal_class.stop_time} s')

            self.problem = self.create_problem(cal_class.tuner_paras)
            samples = self.generate_samples()
            output_array = self.simulate_samples(
                samples=samples,
                cal_class=cal_class)
            result = self.analysis_function(
                x=samples,
                y=output_array
            )
            all_results.append(result)
        return all_results

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
            cal_class.tuner_paras = tuner_paras
        return calibration_classes

    def automatic_select(self, global_res, local_res, calibration_classes):
        # INPUTS
        # 0. Preprocessing
        # 0. Clustering
        # Python-script
        # 1. 1 Klasse, globale SA (mit allen TP) -> Reihenfolge TP
        # 2. X Klassen, lokale SA (mit allen TP) -> Reihenfolge TP
        # Funktion:
        # 3. Vergleich: Finde TP_global[0] == TP_lokal[0]
        # if: nur in einer Klasse -> break
        # if: in mehreren Klassen dominant -> dann wo es ggü dem zweiten am besten ist (max(TP_lokal[0]/TP_lokal[1]))
        # Rekursiv: if: in keiner Klasse -> TP_global[0] == TP_lokal[1]
        # -> 1. zu kalibrierende Klasse mit TP_global[i]
        pass

    def automatic_run(self, calibration_classes):
        # Check input
        calibration_classes = utils.validate_cal_class_input(calibration_classes)
        # Create one global class and run it
        global_classes = calibration_classes.copy()
        # Set the name to global
        for c in global_classes:
            c.name = "global"
        global_res = self.run(global_classes, merge_multiple_classes=True)
        # Run the local analysis
        local_res = self.run(calibration_classes, merge_multiple_classes=True)
        self.automatic_select(global_res,
                              local_res,
                              calibration_classes)
