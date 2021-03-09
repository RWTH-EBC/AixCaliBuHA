from SALib.sample import morris
from SALib.analyze import morris as analyze_morris
from aixcalibuha.sensanalyzer import SensitivityProblem, SenAnalyzer


class MorrisAnalyzer(SenAnalyzer):

    def __init__(self, simulation_api, sensitivity_problem, statistical_measure, **kwargs):
        if not isinstance(sensitivity_problem, MorrisProblem)
        super(MorrisAnalyzer, self).__init__(
            simulation_api=simulation_api,
            sensitivity_problem=sensitivity_problem,
            statistical_measure=statistical_measure,
            **kwargs)

    @property
    def analysis_variables(self):
        return ['mu_star', 'sigma', 'mu_star_conf']

    def analysis_function(self, x, y):
        """
        Use the SALib.analyze.morris method to analyze the simulation results.

        :param np.array x:
            the `X` parameter of the morris method (The NumPy matrix containing the model inputs)
        :param np.array y:
            The NumPy array containing the model outputs
        :return:
            returns the result of the SALib.analyze.sobol method (from the documentation:
            a dictionary with cols `mu`, `mu_star`, `sigma`, and `mu_star_conf`, where each
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
        Run the sampler specified for morris and return the results.

        :return:
            The list of samples generated as a NumPy array with one row per sample
            and each row containing one value for each variable name in `problem['names']`.
        :rtype: np.ndarray
        """
        return morris.sample(self.problem,
                             N=self.sensitivity_problem.num_samples,
                             **self.sensitivity_problem.sampler_parameters)


class MorrisProblem(SensitivityProblem):
    """
    Class for defining relevant information using the morris method.

    See parent class for general attributes.

    **Keyword-arguments:**
    :keyword int num_levels:
        Default 4, used for the morris-method
    :keyword optimal_trajectories:
        Used for the morris-method
    :keyword bool local_optimization:
        Default True, used for the morris-method
    """

    def __init__(self, num_samples, tuner_paras=None, **kwargs):
        """Instantiate instance parameters."""

        self.num_levels = kwargs.pop("num_levels", 4)
        self.optimal_trajectories = kwargs.pop("optimal_trajectories", None)
        self.local_optimization = kwargs.pop("local_optimization", True)

        super(MorrisProblem, self).__init__(method="morris",
                                            num_samples=num_samples,
                                            tuner_paras=tuner_paras,
                                            **kwargs)

    def create_sampler_demand(self):
        """
        Function to create the sampler parameters for the morris method
        """
        return {'num_levels': self.num_levels,
                'optimal_trajectories': self.optimal_trajectories,
                'local_optimization': self.local_optimization}
