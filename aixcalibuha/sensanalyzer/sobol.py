from SALib.sample import saltelli as sobol
from SALib.analyze import sobol as analyze_sobol
from aixcalibuha.sensanalyzer import SenAnalyzer


class SobolAnalyzer(SenAnalyzer):
    """
    Additional arguments:

    **Keyword-arguments:**
    :keyword bool calc_second_order:
        Default True, used for the sobol-method
    :keyword seed:
        Used for the sobol-method
    """
    def __init__(self, sim_api, statistical_measure, **kwargs):
        super().__init__(
            sim_api=sim_api,
            sensitivity_problem=sim_api,
            statistical_measure=statistical_measure,
            **kwargs)
        # Set additional kwargs
        self.calc_second_order = kwargs.pop("calc_second_order", True)
        self.seed = kwargs.pop("seed", None)

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
        return analyze_sobol.analyze(self.problem, y,
                                     calc_second_order=self.calc_second_order)

    def create_sampler_demand(self):
        """
        Function to create the sampler parameters for the morris method
        """
        return {'calc_second_order': self.calc_second_order,
                'seed': self.seed}

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
