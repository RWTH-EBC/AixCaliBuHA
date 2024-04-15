"""
Adds the SobolAnalyzer to the available
classes of sensitivity analysis.
"""
import pandas as pd
from SALib.sample import sobol
from SALib.analyze import sobol as analyze_sobol
import numpy as np
from aixcalibuha.sensitivity_analysis import SenAnalyzer
from aixcalibuha import CalibrationClass
from aixcalibuha.sensitivity_analysis.plotting import plot_single, heatmaps


class SobolAnalyzer(SenAnalyzer):
    """
    Sobol method from SALib
    https://salib.readthedocs.io/en/latest/api.html#sobol-sensitivity-analysis
    A variance-based method which can compute the sensitivity measures
    'S1', 'ST' and 'S2' with their confidence intervals.

    Additional arguments:

    :keyword bool calc_second_order:
        Default True, used for the sobol-method
    :keyword int seed:
        Used for the sobol-method
    """
    __analysis_variables = ['S1', 'ST', 'S1_conf', 'ST_conf']
    __analysis_variables_1 = ['S1', 'ST', 'S1_conf', 'ST_conf']
    __analysis_variables_2 = ['S2', 'S2_conf']

    def __init__(self, sim_api, **kwargs):
        # Additional kwarg which changes the possible analysis_variables
        self.calc_second_order = kwargs.get("calc_second_order", True)
        if self.calc_second_order:
            self.__analysis_variables = ['S1', 'ST', 'S1_conf', 'ST_conf', 'S2', 'S2_conf']
        # separately store first and total order (1) and second order (2) analysis variables
        self.av_1_selected = []
        self.av_2_selected = []
        # Set additional kwargs
        self.seed = kwargs.pop("seed", None)

        super().__init__(
            sim_api=sim_api,
            **kwargs)

    @property
    def analysis_variables(self):
        """The analysis variables of the sobol method"""
        return self.__analysis_variables

    def analysis_function(self, x, y):
        """
        Use the SALib.analyze.sobol method to analyze the simulation results.

        :param np.array x:
            placeholder for the `X` parameter of the morris method not used for sobol
        :param np.array y:
            The NumPy array containing the model outputs
        :return:
            returns the result of the SALib.analyze.sobol method (from the documentation:
            a dictionary with cols `S1`, `S1_conf`, `ST`, and `ST_conf`, where each entry
            is a list of size D (the number of parameters) containing the indices in the same
            order as the parameter file. If calc_second_order is True, the dictionary also
            contains cols `S2` and `S2_conf`.)
        """
        return analyze_sobol.analyze(self.problem, y,
                                     calc_second_order=self.calc_second_order)

    def create_sampler_demand(self):
        """
        Function to create the sampler parameters for the morris method
        """
        return {'calc_second_order': self.calc_second_order}

    def generate_samples(self):
        """
        Run the sampler for sobol and return the results.

        :return:
            The list of samples generated as a NumPy array with one row per sample
            and each row containing one value for each variable name in `problem['names']`.
        :rtype: np.ndarray
        """
        return sobol.sample(self.problem,
                            N=self.num_samples,
                            **self.create_sampler_demand())

    def _save(self, result: tuple, time_dependent: bool = False):
        """
        Save the results of the run and run_time_dependent function of the SobolAnalyzer.
        """
        if not result[0].empty:
            super()._save(result=result[0], time_dependent=time_dependent)
        if time_dependent:
            savepath_result_2 = self.working_directory.joinpath(
                f'{self.__class__.__name__}_results_second_order_time.csv')
        else:
            savepath_result_2 = self.working_directory.joinpath(
                f'{self.__class__.__name__}_results_second_order.csv')
        if not result[1].empty:
            result[1].to_csv(savepath_result_2)
            self.reproduction_files.append(savepath_result_2)

    def _conv_local_results(self, results: list, local_classes: list):
        """
        Convert the result dictionaries form SALib
        of each class and goal into a tuple of two DataFrames.
        First is the single order and second is the second order result.
        If one of the results is not computed an empty list is returned.
        """
        _conv_results = []
        _conv_results_2 = []
        tuples = []
        tuples_2 = []
        for class_results, local_class in zip(results, local_classes):
            for goal, goal_results in class_results.items():
                for analysis_var in self.analysis_variables:
                    res_dict = self._get_res_dict(result=goal_results,
                                                  cal_class=local_class,
                                                  analysis_variable=analysis_var)
                    if analysis_var in self.__analysis_variables_1:
                        _conv_results.append(res_dict)
                        tuples.append((local_class.name, goal, analysis_var))
                    elif analysis_var in self.__analysis_variables_2:
                        for tuner_para, res_dict in res_dict.items():
                            _conv_results_2.append(res_dict)
                            tuples_2.append((local_class.name, goal, analysis_var, tuner_para))
        index = pd.MultiIndex.from_tuples(tuples=tuples,
                                          names=['Class', 'Goal', 'Analysis variable'])
        index_2 = pd.MultiIndex.from_tuples(tuples=tuples_2,
                                            names=['Class', 'Goal', 'Analysis variable',
                                                   'Interaction'])
        df = pd.DataFrame(_conv_results, index=index)
        df_2 = pd.DataFrame(_conv_results_2, index=index_2)
        return df, df_2

    def _get_res_dict(self, result: dict, cal_class: CalibrationClass, analysis_variable: str):
        """
        Convert the result object to a dict with the key
        being the variable name and the value being the result
        associated to analysis_variable.
        For second oder analysis variables the result is converted to a
        dict with the key being the variable name and the value being another dict
        with the variable names as the keys and the result associated to analysis_variable
        from the interaction between the two variables.
        """
        names = self.create_problem(cal_class.tuner_paras)['names']
        if analysis_variable in self.__analysis_variables_1:
            if result is None:
                res_dict_1 = {var_name: np.abs(res_val)
                              for var_name, res_val in zip(names,
                                                           np.zeros(len(names)))}
            else:
                res_dict_1 = {var_name: np.abs(res_val)
                              for var_name, res_val in zip(names,
                                                           result[analysis_variable])}
            return res_dict_1
        if analysis_variable in self.__analysis_variables_2:
            if result is None:
                res_dict_2 = {var_name: dict(zip(names, np.abs(res_val)))
                              for var_name, res_val in zip(names,
                                                           np.zeros((len(names), len(names))))}
            else:
                result_av = result[analysis_variable]
                for i, _ in enumerate(result_av):
                    for j, _ in enumerate(result_av):
                        if i > j:
                            result_av[i][j] = result_av[j][i]
                res_dict_2 = {var_name: dict(zip(names, np.abs(res_val)))
                              for var_name, res_val in zip(names,
                                                           result_av)}
            return res_dict_2

    def plot(self, result):
        """
        Plot the results of the sensitivity analysis method from run().

        :param pd.DataFrame result:
            Dataframe of the results like from the run() function.
        :return tuple of matplotlib objects (fig, ax):
        """
        plot_single(result=result[0])
        heatmaps(result=result[1])

    @staticmethod
    def load_second_order_from_csv(path):
        """
        Load second order sensitivity results which were saved with the run() or
        run_time_dependent() function.
        """
        result = pd.read_csv(path, index_col=[0, 1, 2, 3])
        return result
