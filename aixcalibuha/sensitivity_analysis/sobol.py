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


class SobolAnalyzer(SenAnalyzer):
    """
    Additional arguments:

    :keyword bool calc_second_order:
        Default True, used for the sobol-method
    :keyword seed:
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
        # seperatly store first and total oder (1) and second order (2) analysis variables
        self.av_1_selected = []
        self.av_2_selected = []

        super().__init__(
            sim_api=sim_api,
            **kwargs)
        # Set additional kwargs
        self.seed = kwargs.pop("seed", None)



    @property
    def analysis_variables(self):
        """The analysis variables of the sobol method"""
        return self.__analysis_variables

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
                continue
            if v in self.__analysis_variables_1:
                self.av_1_selected.append(v)
                continue
            if v in self.__analysis_variables_2:
                self.av_2_selected.append(v)
        if false_values:
            error_message = f'Given analysis_variable "{false_values}" not ' \
                            f'supported for class {self.__class__.__name__}. ' \
                            f'Supported options are: {", ".join(self.analysis_variables)}.'
            if not self.calc_second_order:
                error_message += f' When calc_second_order also S2 und S2_conf are supported.'
            raise ValueError(error_message)
        self._analysis_variable = value

    def analysis_function(self, x, y):
        """
        Use the SALib.analyze.sobol method to analyze the simulation results.

        :param np.array: x
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

    def _conv_local_results(self, results: list, local_classes: list, verbose=False):
        _conv_results = []
        _conv_restuls_2 = []
        tuples = []
        tuples_2 = []
        for class_results, local_class in zip(results, local_classes):
            for goal, goal_results in class_results.items():
                for av in self.analysis_variable:
                    res_dict = self._get_res_dict(result=goal_results,
                                                  cal_class=local_class,
                                                  analysis_variable=av)
                    if av in self.__analysis_variables_1:
                        _conv_results.append(res_dict)
                        tuples.append((local_class.name, goal, av))
                    elif av in self.__analysis_variables_2:
                        for tuner_para, res_dict in res_dict.items():
                            _conv_restuls_2.append(res_dict)
                            tuples_2.append((local_class.name, goal, av, tuner_para))
        index = pd.MultiIndex.from_tuples(tuples=tuples,
                                          names=['Class', 'Goal', 'Analysis variable'])
        index_2 = pd.MultiIndex.from_tuples(tuples=tuples_2,
                                            names=['Class', 'Goal', 'Analysis variable', 'Interaction'])
        df = pd.DataFrame(_conv_results, index=index)
        df_2 = pd.DataFrame(_conv_restuls_2, index=index_2)
        return (df, df_2)

    def _get_res_dict(self, result: dict, cal_class: CalibrationClass, analysis_variable: str):
        """
        Convert the result object to a dict with the key
        being the variable name and the value being the result
        associated to analysis_variable.
        For second oder analyisis variables the result is convertet to a
        dict with the key being the variable name and the value being another dict
        with the vaiable names as the keys and the result associated to analysis_valiable
        from the interaction between the two variables.
        """
        #res_dict = {'res_dict_1': None, 'res_dict_2': None}
        names = self.create_problem(cal_class.tuner_paras)['names']
        if analysis_variable in self.__analysis_variables_1:
            res_dict_1 = {var_name: np.abs(res_val)
                          for var_name, res_val in zip(names,
                                                       result[analysis_variable])}
            return res_dict_1
        if analysis_variable in self.__analysis_variables_2:
            result_av = result[analysis_variable]
            for i in range(len(result_av)):
                for j in range(len(result_av)):
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
        :return tuple of matplotlib objects (fig, ax)
        """
        self.plot_single(result=result[0])


