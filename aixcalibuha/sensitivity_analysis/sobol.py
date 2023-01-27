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
import matplotlib.pyplot as plt


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
        # separately store first and total order (1) and second order (2) analysis variables
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

    def info_samples(self, cal_class, scale):
        """
        Saves an info.txt about the configuration of the SenAnalyser for the creation of the samples
        if the simulation files and samples are saved.
        """
        with open(self.savepath_sim.joinpath(f'info_{cal_class.name}.txt'), 'w') as f:
            f.write(f'Configuration SenAnalyser:\n'
                    f'SenAnalyser: {self.__class__.__name__}\n'
                    f'Logger: {self.cd.joinpath(self.__class__.__name__)}\n'
                    f'num_samples: {self.num_samples}\n'
                    f'calc_second_order: {self.calc_second_order}\n'
                    f'scale: {scale}\n'
                    f'Model: {self.sim_api.model_name}\n'
                    f'Tuner-Paras:\n'
                    f'{cal_class.tuner_paras._df.to_string()}')

    def _save(self, result):
        if not result[0].empty:
            result[0].to_csv(self.cd.joinpath(f'{self.__class__.__name__}_results.csv'))
        if not result[1].empty:
            result[1].to_csv(self.cd.joinpath(f'{self.__class__.__name__}_results_second_order.csv'))

    def _conv_local_results(self, results: list, local_classes: list, verbose=False):
        _conv_results = []
        _conv_results_2 = []
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
                            _conv_results_2.append(res_dict)
                            tuples_2.append((local_class.name, goal, av, tuner_para))
        index = pd.MultiIndex.from_tuples(tuples=tuples,
                                          names=['Class', 'Goal', 'Analysis variable'])
        index_2 = pd.MultiIndex.from_tuples(tuples=tuples_2,
                                            names=['Class', 'Goal', 'Analysis variable', 'Interaction'])
        df = pd.DataFrame(_conv_results, index=index)
        df_2 = pd.DataFrame(_conv_results_2, index=index_2)
        result = (df, df_2)
        return result

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

    @staticmethod
    def plot_second_order(result: pd.DataFrame, **kwargs):
        """
        Plot sensitivity results of second order analysis variables.
        For each calibration class and goal one figure of a 3d plot is created
        with the barplots of the interactions for each parameter.
        Only working for more than 2 parameter.

        :param pd.DataFrame result:
            A result from run
        :keyword bool show_plot:
            Default is True. If False, all created plots are not shown.
        :keyword bool use_suffix:
            Default is True: If True, the last part after the last point
            of Modelica variables is used for the x ticks.
        :return:
            Returns all created figures and axes in lists like [fig], [ax]
        """
        show_plot = kwargs.pop('show_plot', True)
        # kwargs for the design
        use_suffix = kwargs.pop('use_suffix', False)
        result = result.fillna(0)
        # get lists of the calibration classes their goals and the analysis variables in the result dataframe
        cal_classes = SenAnalyzer._del_duplicates(list(result.index.get_level_values(0)))
        goals = SenAnalyzer._del_duplicates(list(result.index.get_level_values(1)))

        # rename tuner_names in result to the suffix of their variable name
        if use_suffix:
            result = SenAnalyzer._rename_tuner_names(result)
            print('Second order results with suffixes of tuner-parameter names:')
            print(result.to_string())

        tuner_names = result.columns
        if len(tuner_names) < 3:
            return None
        xticks = np.arange(len(tuner_names))

        # when the index is not sorted pandas throws a performance warning
        result = result.sort_index()

        # plot of S2 without S2_conf
        all_figs = []
        all_axes = []
        for cal_class in cal_classes:
            class_figs = []
            class_axes = []
            for goal in goals:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                for idx, name in enumerate(tuner_names):
                    ax.bar(tuner_names, result.loc[cal_class, goal, 'S2', name].to_numpy(), zs=idx, zdir='y', alpha=0.8)
                    ax.set_title(f"{cal_class} {goal}")
                    ax.set_zlabel('S2 [-]')
                    ax.set_yticks(xticks)
                    ax.set_yticklabels(tuner_names)
                    # rotate tick labels for better readability
                    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
                    plt.setp(ax.get_yticklabels(), rotation=90, ha="right", rotation_mode="anchor")
                class_figs.append(fig)
                class_axes.append(ax)
            all_figs.append(class_figs)
            all_axes.append(class_axes)
        if show_plot:
            plt.show()
        return all_figs, all_axes

    @staticmethod
    def plot_single_second_order(result, para_name, **kwargs):
        """
        Plot the value of S2 from one parameter with all other parameters.

        :param pd.DataFrame result:
            Second order result from run.
        :param str para_name:
            Name of the parameter of which the results should be plotted.
        :keyword bool show_plot:
            Default is True. If False, all created plots are not shown.
        :return:
            Returns all created figures and axes in lists like [fig], [ax]
        """
        show_plot = kwargs.pop('show_plot', True)
        result = result.loc[:, :, :, para_name][:].fillna(0)
        figs, axes = SenAnalyzer.plot_single(result=result, show_plot=False)
        # set new title for the figures of each calibration class
        for fig in figs:
            fig.suptitle(f"Interaction of {para_name} in class {fig._suptitle.get_text()}")
        if show_plot:
            plt.show()
        return figs, axes

    def plot(self, result):
        """
        Plot the results of the sensitivity analysis method from run().

        :param pd.DataFrame result:
            Dataframe of the results like from the run() function.
        :return tuple of matplotlib objects (fig, ax)
        """
        SobolAnalyzer.plot_single(result=result[0])
        SobolAnalyzer.plot_second_order(result=result[1])

    @staticmethod
    def load_second_order_from_csv(path):
        result = pd.read_csv(path, index_col=[0, 1, 2, 3])
        return result
