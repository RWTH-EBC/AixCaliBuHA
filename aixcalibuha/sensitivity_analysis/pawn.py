"""
Adds the PAWNAnalyzer to the available
classes of sensitivity analysis.
"""

from SALib.sample import sobol
from SALib.sample import morris
from SALib.sample import fast_sampler as fast
from SALib.analyze import pawn as analyze_pawn
import numpy as np
from aixcalibuha.sensitivity_analysis import SenAnalyzer
from aixcalibuha import CalibrationClass


class PAWNAnalyzer(SenAnalyzer):
    """
    PAWN method from SALib https://salib.readthedocs.io/en/latest/api.html#pawn-sensitivity-analysis
    Density-based method which computes the PAWN index at 'min', 'max', 'mean',
    'median' and coefficient of variation 'cv'.
    
    Additional arguments:

    :keyword bool calc_second_order:
        Default True, used for the sampler of the sobol-method
    :keyword int s:
        Default 10, used for the pawn-method.
    :keyword str sampler:
        Which sampler should be used. Default sobol.
        Choose between 'sobol', 'morris' and 'fast'.
    :keyword int num_levels:
        Default num_samples, used for the sampler of the morris-method.
    :keyword optimal_trajectories:
        Used for the sampler of the morris-method.
    :keyword bool local_optimization:
        Default True, used for the sampler of the morris-method.
    :keyword int M:
        Default 4, used for the sampler of the fast-method.
    """

    def __init__(self, sim_api, **kwargs):
        super().__init__(
            sim_api=sim_api,
            **kwargs)
        # Set additional kwargs
        self.calc_second_order = kwargs.pop("calc_second_order", True)
        self.s = kwargs.pop("s", 10)
        self.sampler = kwargs.pop("sampler", 'sobol')
        self.num_levels = kwargs.pop("num_levels", self.num_samples)
        self.optimal_trajectories = kwargs.pop("optimal_trajectories", None)
        self.local_optimization = kwargs.pop("local_optimization", True)
        self.M = kwargs.pop("M", 4)

    @property
    def analysis_variables(self):
        """The analysis variables of the PAWN method"""
        return ['minimum', 'mean', 'median', 'maximum', 'CV']

    def analysis_function(self, x, y):
        """
        Use the SALib.analyze.pawn method to analyze the simulation results.

        :param np.array x:
            placeholder for the `X` parameter of the morris method not used for sobol
        :param np.array y:
            The NumPy array containing the model outputs
        :return:
            returns the result of the SALib.analyze.pawn method (from the documentation:
            This implementation reports the PAWN index at the min, mean, median, and
            max across the slides/conditioning intervals as well as the coefficient of
            variation (``CV``). The median value is the typically reported value. As
            the ``CV`` is (standard deviation / mean), it indicates the level of
            variability across the slides, with values closer to zero indicating lower
            variation.)
        """
        return analyze_pawn.analyze(self.problem, x, y,
                                    S=self.s)

    def create_sampler_demand(self):
        """
        Function to create the sampler parameters for the sampler method
        """
        if self.sampler == 'sobol':
            return {'calc_second_order': self.calc_second_order}
        elif self.sampler == 'morris':
            return {'num_levels': self.num_levels,
                    'optimal_trajectories': self.optimal_trajectories,
                    'local_optimization': self.local_optimization}
        elif self.sampler == 'fast':
            return {'M': self.M}
        else:
            raise NotImplementedError(f'{self.sampler} is not implemented yet')

    def generate_samples(self):
        """
        Run the sampler for the selected sampler and return the results.

        :return:
            The list of samples generated as a NumPy array with one row per sample
            and each row containing one value for each variable name in `problem['names']`.
        :rtype: np.ndarray
        """
        if self.sampler == 'sobol':
            return sobol.sample(self.problem,
                                N=self.num_samples,
                                **self.create_sampler_demand())
        if self.sampler == 'morris':
            return morris.sample(self.problem,
                                 N=self.num_samples,
                                 **self.create_sampler_demand())
        if self.sampler == 'fast':
            return fast.sample(self.problem,
                               N=self.num_samples,
                               **self.create_sampler_demand())
        else:
            raise NotImplementedError(f'{self.sampler} is not implemented yet')

    def _get_res_dict(self, result: dict, cal_class: CalibrationClass, analysis_variable: str):
        """
        Convert the result object to a dict with the key
        being the variable name and the value being the result
        associated to self.analysis_variable.
        """
        if result is None:
            names = cal_class.tuner_paras.get_names()
            return dict(zip(names, np.zeros(len(names))))
        return dict(zip(result['names'], result[analysis_variable]))
