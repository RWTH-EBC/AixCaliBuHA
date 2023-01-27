"""
Adds the MorrisAnalyzer to the available
classes of sensitivity analysis.
"""
import warnings

from SALib.sample import morris
from SALib.analyze import morris as analyze_morris
from aixcalibuha.sensitivity_analysis import SenAnalyzer
from aixcalibuha import CalibrationClass


class MorrisAnalyzer(SenAnalyzer):
    """
    Additional arguments:

    :keyword int num_levels:
        Default num_samples, used for the morris-method
    :keyword optimal_trajectories:
        Used for the morris-method
    :keyword bool local_optimization:
        Default True, used for the morris-method
    """
    def __init__(self, sim_api, **kwargs):
        super().__init__(
            sim_api=sim_api,
            **kwargs)
        # Set additional kwargs
        self.num_levels = kwargs.pop("num_levels", self.num_samples)
        self.optimal_trajectories = kwargs.pop("optimal_trajectories", None)
        self.local_optimization = kwargs.pop("local_optimization", True)

    @property
    def analysis_variables(self):
        """The analysis variables of the sobol method"""
        return ['mu_star', 'mu', 'sigma', 'mu_star_conf']

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
        return analyze_morris.analyze(self.problem, x, y,
                                      num_levels=self.num_levels)

    def create_sampler_demand(self):
        """
        Function to create the sampler parameters for the morris method
        """
        return {'num_levels': self.num_levels,
                'optimal_trajectories': self.optimal_trajectories,
                'local_optimization': self.local_optimization}

    def generate_samples(self):
        """
        Run the sampler specified for morris and return the results.

        :return:
            The list of samples generated as a NumPy array with one row per sample
            and each row containing one value for each variable name in `problem['names']`.
        :rtype: np.ndarray
        """
        return morris.sample(self.problem,
                             N=self.num_samples,
                             **self.create_sampler_demand())

    def info_samples(self, cal_class, scale):
        """
        Saves an info.txt about the configuration of the SenAnalyser for the creation of the samples
        if the simulation files and samples are saved-
        """
        with open(self.savepath_sim.joinpath(f'info_{cal_class.name}.txt'), 'w') as f:
            f.write(f'Configuration SenAnalyser:\n'
                    f'SenAnalyser: {self.__class__.__name__}\n'
                    f'Logger: {self.cd.joinpath(self.__class__.__name__)}\n'
                    f'num_samples: {self.num_samples}\n'
                    f'num_levels: {self.num_levels}\n'
                    f'local_optimization: {self.local_optimization}\n'
                    f'scale: {scale}\n'
                    f'Model: {self.sim_api.model_name}\n'
                    f'Tuner-Paras:\n'
                    f'{cal_class.tuner_paras._df.to_string()}')

    def _get_res_dict(self, result: dict, cal_class: CalibrationClass, analysis_variable: str):
        """
        Convert the result object to a dict with the key
        being the variable name and the value being the result
        associated to self.analysis_variable.
        """
        return dict(zip(result['names'], result[analysis_variable]))
