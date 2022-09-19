"""
Adds the FASTAnalyzer to the available
classes of sensitivity analysis.
"""
from SALib.sample import fast_sampler as fast
from SALib.analyze import fast as analyze_fast
import numpy as np
from aixcalibuha.sensitivity_analysis import SenAnalyzer
from aixcalibuha import CalibrationClass


class FASTAnalyzer(SenAnalyzer):
    """
    Additional arguments:

    :keyword int M:
        Default 4, used for the fast-method
    :keyword seed:
        Used for the fast-method
    """

    def __init__(self, sim_api, **kwargs):
        super().__init__(
            sim_api=sim_api,
            **kwargs)
        # Set additional kwargs
        self.M = kwargs.pop("M", 4)
        self.seed = kwargs.pop("seed", None)

    @property
    def analysis_variables(self):
        """The analysis variables of the FAST method"""
        return ['S1', 'ST']

    def analysis_function(self, x, y):
        """
        Use the SALib.analyze.fast method to analyze the simulation results.

        :param np.array: x
            placeholder for the `X` parameter of the morris method not used for sobol
        :param np.array y:
            The NumPy array containing the model outputs
        :return:
            returns the result of the SALib.analyze.fast method (from the documentation:
            Returns a dictionary with keys 'S1' and 'ST', where each entry is a list of
            size D (the number of parameters) containing the indices in the same order
            as the parameter file.)
        """
        return analyze_fast.analyze(self.problem, y,
                                    M=self.M)

    def create_sampler_demand(self):
        """
        Function to create the sampler parameters for the fast method
        """
        return {'M': self.M}

    def generate_samples(self):
        """
        Run the sampler for fast and return the results.

        :return:
            The list of samples generated as a NumPy array with one row per sample
            and each row containing one value for each variable name in `problem['names']`.
        :rtype: np.ndarray
        """
        return fast.sample(self.problem,
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
