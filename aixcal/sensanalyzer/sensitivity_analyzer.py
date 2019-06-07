"""
Module for classes revolving around performing sensitivity analysis.
"""

import copy
from SALib.sample import morris
from SALib.sample import saltelli as sobol
from SALib.analyze import morris as analyze_morris
from SALib.analyze import sobol as analyze_sobol
from aixcal.utils import visualizer
from aixcal import data_types
from aixcal import simulationapi
import numpy as np


class SenAnalyzer:
    """
    Class to perform a Sensitivity Analysis.
    :param cd: str, os.path.normpath
        The path for the current working directory.
    :type cd: str
    :param: simulation_api
        Simulation-API used to simulate the samples
    :param: sensitivity_problem
        Parameter class for the sensitivity. it contains the demand of the sampler
        and create the dictionary parameter Problem
    :param calibration_classes: list, data_types.CalibrationClass
        Either one or multiple classes for calibration
    :param statistical_measure:
        Used to evaluate the difference of simulated and measured data.
        Like "RMSE", "MAE" etc. See utils.statistics_analyzer.py for
        further info.
    :type statistical_measure: str
    """
    simulation_api = simulationapi.SimulationAPI
    tuner_paras = data_types.TunerParas
    goals = data_types.Goals

    def __init__(self, cd, simulation_api, sensitivity_problem,
                 calibration_classes, statistical_measure):
        """Instantiate class parameters"""
        # Setup the logger
        self.logger = visualizer.Logger(cd, self.__class__.__name__)
        # Add any simulation_api, dymolapi or pyfmi
        self.simulation_api = simulation_api
        self.statistical_measure = statistical_measure

        self.sensitivity_problem = sensitivity_problem
        self.problem = sensitivity_problem.problem
        self.method = sensitivity_problem.method

        # Check correct input of parameters.
        if isinstance(calibration_classes, list):
            for cal_class in calibration_classes:
                if not isinstance(cal_class, data_types.CalibrationClass):
                    raise TypeError("calibration_classes is of type {} but should "
                                    "be {}".format(type(cal_class).__name__,
                                                   type(data_types.CalibrationClass).__name__))
            self.calibration_classes = calibration_classes
        elif isinstance(calibration_classes, data_types.CalibrationClass):
            self.calibration_classes = [calibration_classes]
        else:
            raise TypeError("calibration_classes is of type {} but should "
                            "be {} or list".format(type(calibration_classes).__name__,
                                                   type(data_types.CalibrationClass).__name__))

        # Choose which analysis function to use and the list of keys in the analysis output to store
        if self.method.lower() == 'morris':
            self.analysis_function = self.morris_analyze_function
            self.analysis_variables = ['mu_star', 'sigma', 'mu_star_conf']
        elif self.method.lower() == 'sobol':
            self.analysis_function = self.sobol_analyze_function
            self.analysis_variables = ['S1', 'ST', 'ST_conf']
        else:
            raise ValueError("Invalid analysis method: %s" % self.method)

    def sobol_analyze_function(self, _, y):
        """
        Use the SALib.analyze.sobol method to analyze the simulation results.
        :param _: None
            placeholder for the `X` parameter of the morris method not used for sobol
        :param y: np.array
            The NumPy array containing the model outputs
        :return:
            returns the result of the SALib.analyze.sobol method (from the documentation:
            a dictionary with keys `S1`, `S1_conf`, `ST`, and `ST_conf`, where each entry
            is a list of size D (the number of parameters) containing the indices in the same
            order as the parameter file. If calc_second_order is True, the dictionary also
            contains keys `S2` and `S2_conf`.)
        """
        if 'calc_second_order' not in self.sensitivity_problem.sampler_parameters:
            raise KeyError('sobol method requires the `calc_second_order`'
                           ' parameter to be set (bool)')
        calc_second_order = self.sensitivity_problem.sampler_parameters['calc_second_order']
        return analyze_sobol.analyze(self.problem, y,
                                     calc_second_order=calc_second_order)

    def morris_analyze_function(self, x, y):
        """
        Use the SALib.analyze.morris method to analyze the simulation results.
        :param x: np.array
            the `X` parameter of the morris method (The NumPy matrix containing the model inputs)
        :param y: np.array
            The NumPy array containing the model outputs
        :return:
            returns the result of the SALib.analyze.sobol method (from the documentation:
            a dictionary with keys `mu`, `mu_star`, `sigma`, and `mu_star_conf`, where each
            entry is a list of size D (the number of parameters) containing the indices in the
            same order as the parameter file.)
        """
        if 'num_levels' not in self.sensitivity_problem.sampler_parameters:
            raise KeyError('morris method requires the `num_levels` parameter to be set (int)')
        num_levels = self.sensitivity_problem.sampler_parameters['num_levels']
        return analyze_morris.analyze(self.problem, x, y,
                                      num_levels=num_levels)

    def generate_samples(self):
        """
        Run the sampler specified by `method` and return the results.
        :return:
            The list of samples generated as a NumPy array with one row per sample
            and each row containing one value for each variable name in `problem['names']`.
        :rtype: np.ndarray
        """
        if self.method == 'morris':
            return morris.sample(self.problem,
                                 N=self.sensitivity_problem.num_samples,
                                 **self.sensitivity_problem.sampler_parameters)
        elif self.method == 'sobol':
            return sobol.sample(self.problem,
                                N=self.sensitivity_problem.num_samples,
                                **self.sensitivity_problem.sampler_parameters)
        else:
            raise ValueError("Sampler method unknown: %s" % self.method)

    def simulate_samples(self, samples, start_time, stop_time):
        """
        Put the parameter in dymola model, run it.
        :param:
         Output variables in dymola
        :param: model_input_SA
        generated sample data as input to simulation
        :return: np.array
            An array containing the evaluated differences for each sample
        """
        output = []
        initial_names = self.tuner_paras.get_names()
        self.simulation_api.set_sim_setup({"initialNames": initial_names,
                                           "startTime": start_time,
                                           "stopTime": stop_time})
        for i, initial_values in enumerate(samples):
            # Simulate the current values
            self.logger.log('Parameter variation {} of {}'.format(i+1, len(samples)))
            self.simulation_api.set_initial_values(initial_values)
            filepath = self.simulation_api.simulate()

            # Load the result file to the goals object
            sim_target_data = data_types.SimTargetData(filepath)
            self.goals.set_sim_target_data(sim_target_data)
            self.goals.set_relevant_time_interval(start_time,
                                                  stop_time)

            # Evaluate the current objective
            total_res = self.goals.eval_difference(self.statistical_measure)
            output.append(total_res)

        return np.asarray(output)

    def run(self):
        """
        Execute the sensitivity analysis for each class and
        return the result.
        :return:
            Returns a list of dictionaries. One dict is the SALib-result for
            one calibration-class. The order is based on the order of the
            calibration-class list.
        :rtype: list
        """
        all_results = []
        for cal_class in self.calibration_classes:
            self.logger.log('Start sensitivity analysis of class: {}, '
                            'Time-Interval: {}-{} s'.format(cal_class.name,
                                                            cal_class.start_time,
                                                            cal_class.stop_time))
            self.tuner_paras = cal_class.tuner_paras
            self.goals = cal_class.goals
            self.problem = SensitivityProblem.create_problem(self.tuner_paras)
            samples = self.generate_samples()
            output_array = self.simulate_samples(samples,
                                                 start_time=cal_class.start_time,
                                                 stop_time=cal_class.stop_time)
            salib_analyze_result = self.analysis_function(samples, output_array)
            all_results.append(salib_analyze_result)
        return all_results

    @staticmethod
    def automatic_select(calibration_classes, result, threshold, key="mu_star"):
        """
        Automatically select sensitive tuner parameters based on a given threshold
        and a key-word of the result.
        :param calibration_classes:
            List of data_types.CalibrationClass objects that you want to
            automatically select sensitive tuner-parameters.
        :type calibration_classes: list
        :param result:
            List of dicts (Sensitivity results)
        :type result: list
        :param threshold: Minimal required value of given key
        :type threshold: float
        :param key:
            Value that is used to define the sensitivity.
            Default is mu_star, "the absolute mean elementary effect"
            Choose between: mu, mu_star, sigma, mu_star_conf
        :type key: str
        :return:
        """
        for num_class, cal_class in enumerate(calibration_classes):
            class_result = result[num_class]
            tuner_paras = copy.deepcopy(cal_class.tuner_paras)
            select_names = []
            for i, sen_value in enumerate(class_result[key]):
                if sen_value < threshold:
                    select_names.append(class_result["names"][i])
            tuner_paras.remove_names(select_names)
            cal_class.set_tuner_paras(tuner_paras)
        return calibration_classes


class SensitivityProblem:
    """
    Class for defining relevant
    :param method: The method to use. Valid values are 'morris' (default) and 'sobol'.
    :type method: str
    :param method: The method to use. Valid values: 'morris', 'sobol'
    :type method: str
    :param num_samples:
        The parameter `N` to the sampler methods of sobol and morris. NOTE: This is not the
        the number of samples produced, but relates to the total number of samples produced in
        a manner dependent on the sampler method used. See the documentation of sobol and
        morris in the SALib for more information.
    :type num_samples: int
    """
    def __init__(self, method, num_samples, tuner_paras=None):
        """Instantiate instance parameters"""
        self.method = method
        self.num_samples = num_samples
        self.sampler_parameters = self.create_sampler_demand()
        if tuner_paras is not None:
            self.problem = self.create_problem(tuner_paras)
        else:
            self.problem = None

    def create_sampler_demand(self, calc_second_order=True, seed=None, num_levels=4,
                              optimal_trajectories=None, local_optimization=True):
        """
        Function to create the sampler parameters for each different method of
        sensitivity analysis.
        :param calc_second_order:
        :param seed:
        :param num_levels:
        :param optimal_trajectories:
        :param local_optimization:
        :return:
        """
        if self.method == 'morris':
            sampler_parameters = {'num_levels': num_levels,
                                  'optimal_trajectories': optimal_trajectories,
                                  'local_optimization': local_optimization}
        elif self.method == 'sobol':
            sampler_parameters = {'calc_second_order': calc_second_order,
                                  'seed': seed}
        else:
            raise KeyError("Given method {} is not supported.".format(self.method))
        return sampler_parameters

    @staticmethod
    def create_problem(tuner_paras):
        """Create function for later access if multiple calibration-classes are used """
        num_vars = len(tuner_paras.get_names())
        bounds = np.array(tuner_paras.get_bounds())
        problem = {'num_vars': num_vars,
                   'names': tuner_paras.get_names(),
                   'bounds': np.transpose(bounds)}
        return problem
