"""
Adds the SobolAnalyzer to the available
classes of sensitivity analysis.
"""
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
    __sobol_analysis_variables = ['S1', 'ST', 'S1_conf', 'ST_conf']

    def __init__(self, sim_api, **kwargs):
        # Additional kwarg which changes the possible analysis_variables
        self.calc_second_order = kwargs.get("calc_second_order", True)
        if self.calc_second_order:
            self.__sobol_analysis_variables = ['S1', 'ST', 'S1_conf', 'ST_conf', 'S2', 'S2_conf']

        super().__init__(
            sim_api=sim_api,
            **kwargs)
        # Set additional kwargs
        self.seed = kwargs.pop("seed", None)

    @property
    def analysis_variables(self):
        """The analysis variables of the sobol method"""
        return self.__sobol_analysis_variables

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

    def _get_res_dict(self, result: dict, cal_class: CalibrationClass, analysis_variable: str):
        """
        Convert the result object to a dict with the key
        being the variable name and the value being the result
        associated to self.analysis_variable.
        """
        names = self.create_problem(cal_class.tuner_paras)['names']
        return {var_name: np.abs(res_val)
                for var_name, res_val in zip(names,
                                             result[analysis_variable])}
