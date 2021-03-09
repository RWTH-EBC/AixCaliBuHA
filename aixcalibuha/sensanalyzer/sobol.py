from SALib.sample import saltelli as sobol
from SALib.analyze import sobol as analyze_sobol
from aixcalibuha.sensanalyzer import SensitivityProblem, SenAnalyzer


class SobolAnalyzer(SenAnalyzer):

    def __init__(self, simulation_api, sensitivity_problem, statistical_measure, **kwargs):
        super(SobolAnalyzer, self).__init__(
            simulation_api=simulation_api,
            sensitivity_problem=sensitivity_problem,
            statistical_measure=statistical_measure,
            **kwargs)

    @property
    def analysis_variables(self):
        return ['S1', 'ST', 'ST_conf']

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
        if 'calc_second_order' not in self.sensitivity_problem.sampler_parameters:
            raise KeyError('sobol method requires the `calc_second_order`'
                           ' parameter to be set (bool)')
        calc_second_order = self.sensitivity_problem.sampler_parameters['calc_second_order']
        return analyze_sobol.analyze(self.problem, y,
                                     calc_second_order=calc_second_order)

    def generate_samples(self):
        """
        Run the sampler for sobol and return the results.

        :return:
            The list of samples generated as a NumPy array with one row per sample
            and each row containing one value for each variable name in `problem['names']`.
        :rtype: np.ndarray
        """
        return sobol.sample(self.problem,
                            N=self.sensitivity_problem.num_samples,
                            **self.sensitivity_problem.sampler_parameters)


class SobolProblem(SensitivityProblem):
    """
    Class for defining relevant information for performing a sensitivity analysis.

    See parent class for general attributes.

    **Keyword-arguments:**
    :keyword bool calc_second_order:
        Default True, used for the sobol-method
    :keyword seed:
        Used for the sobol-method
    """

    def __init__(self, num_samples, tuner_paras=None, **kwargs):
        """Instantiate instance parameters."""

        self.calc_second_order = kwargs.pop("calc_second_order", True)
        self.seed = kwargs.pop("seed", None)

        super(SobolProblem, self).__init__(method="sobol",
                                           num_samples=num_samples,
                                           tuner_paras=tuner_paras,
                                           **kwargs)

    def create_sampler_demand(self):
        """
        Function to create the sampler parameters for the morris method
        """
        return {'calc_second_order': self.calc_second_order,
                'seed': self.seed}
