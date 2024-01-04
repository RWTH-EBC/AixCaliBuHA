"""
Adds the SobolAnalyzer to the available
classes of sensitivity analysis.
"""
import pandas as pd
from SALib.sample import sobol
from SALib.analyze import sobol as analyze_sobol
import numpy as np
from aixcalibuha.sensitivity_analysis import SenAnalyzer, _del_duplicates, _rename_tuner_names
from aixcalibuha import CalibrationClass
import matplotlib.pyplot as plt


class SobolAnalyzer(SenAnalyzer):
    """
    Sobol method from SALib https://salib.readthedocs.io/en/latest/api.html#sobol-sensitivity-analysis
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

    def _save(self, result, time_dependent=False):
        if time_dependent:
            savepath_result_1 = self.cd.joinpath(f'{self.__class__.__name__}_results_time.csv')
            savepath_result_2 = self.cd.joinpath(f'{self.__class__.__name__}_results_second_order_time.csv')
        else:
            savepath_result_1 = self.cd.joinpath(f'{self.__class__.__name__}_results.csv')
            savepath_result_2 = self.cd.joinpath(f'{self.__class__.__name__}_results_second_order.csv')
        if not result[0].empty:
            result[0].to_csv(savepath_result_1)
            self.reproduction_files.append(savepath_result_1)
        if not result[1].empty:
            result[1].to_csv(savepath_result_2)
            self.reproduction_files.append(savepath_result_2)

    def _conv_local_results(self, results: list, local_classes: list, verbose=False):
        _conv_results = []
        _conv_results_2 = []
        tuples = []
        tuples_2 = []
        for class_results, local_class in zip(results, local_classes):
            for goal, goal_results in class_results.items():
                for av in self.analysis_variables:
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
        :keyword [str] cal_classes:
            Default are all possible calibration classes. If a list of
            names of calibration classes is given only plots for these
            classes are created.
        :keyword [str] goals:
            Default are all possible goal names. If a list of specific
            goal names is given only these will be plotted.
        :keyword [[fig]] figs:
            Default None. Useful for using subfigures (see example for verbose sensitivity analysis).
        :return:
            Returns all created figures and axes in lists like [fig], [ax]
        """
        show_plot = kwargs.pop('show_plot', True)
        # kwargs for the design
        use_suffix = kwargs.pop('use_suffix', False)
        figs = kwargs.pop('figs', None)
        result = result.fillna(0)
        # get lists of the calibration classes and their goals in the result dataframe
        cal_classes = _del_duplicates(list(result.index.get_level_values(0)))
        goals = _del_duplicates(list(result.index.get_level_values(1)))
        cal_classes = kwargs.pop('cal_classes', cal_classes)
        goals = kwargs.pop('goals', goals)

        # rename tuner_names in result to the suffix of their variable name
        if use_suffix:
            result = _rename_tuner_names(result)

        tuner_names = result.columns
        if len(tuner_names) < 2:
            return None
        xticks = np.arange(len(tuner_names))

        # when the index is not sorted pandas throws a performance warning
        result = result.sort_index()

        # plot of S2 without S2_conf
        all_figs = []
        all_axes = []
        for class_idx, cal_class in enumerate(cal_classes):
            class_figs = []
            class_axes = []
            for goal_idx, goal in enumerate(goals):
                if figs is None:
                    fig = plt.figure()
                else:
                    fig = figs[class_idx][goal_idx]
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
        :keyword [str] cal_classes:
            Default are all possible calibration classes. If a list of
            names of calibration classes is given only plots for these
            classes are created.
        :keyword [str] goals:
            Default are all possible goal names. If a list of specific
            goal names is given only these will be plotted.
        :keyword bool show_plot:
            Default is True. If False, all created plots are not shown.
        :keyword ([fig], [ax]) axes:
            Default None. Useful for using subfigures (see example for verbose sensitivity analysis).
        :return:
            Returns all created figures and axes in lists like [fig], [ax]
        """
        cal_classes = kwargs.pop('cal_classes', None)
        goals = kwargs.pop('goals', None)
        figs_axes = kwargs.pop('figs_axes', None)
        show_plot = kwargs.pop('show_plot', True)
        use_suffix = kwargs.pop('use_suffix', False)
        result = result.loc[:, :, :, para_name][:].fillna(0)
        figs, axes = SenAnalyzer.plot_single(
            result=result,
            show_plot=False,
            cal_classes=cal_classes,
            goals=goals,
            figs_axes=figs_axes,
            use_suffix=use_suffix
        )
        # set new title for the figures of each calibration class
        for fig in figs:
            fig.suptitle(f"Interaction of {para_name} in class {fig._suptitle.get_text()}")
        if show_plot:
            plt.show()
        return figs, axes

    @staticmethod
    def heatmap(result, cal_class, goal, ax=None, show_plot=True, use_suffix=False):
        """
        Plot S2 sensitivity results from one calibration class and goal as a heatmap.

        :param pd.DataFrame result:
            A second order result from run
        :param str cal_class:
            Name of the class to plot S2 from.
        :param str goal:
            Name of the goal to plot S2 from.
        :param matplotlib.axes ax:
            Default is None. If an axes is given the heatmap will be plotted on it else
            a new figure and axes is created.
        :param bool show_plot:
            Default is True. If False, all created plots are not shown.
        :param bool use_suffix:
            Default is False. If True, only the last suffix of a Modelica variable is displayed.
        :return:
            Returns axes
        """
        if use_suffix:
            result = _rename_tuner_names(result)
        if ax is None:
            fig, ax = plt.subplots()
        data = result.sort_index().loc[cal_class, goal, 'S2'].fillna(0).reindex(
            index=result.columns)
        im = ax.imshow(data, cmap='Reds')
        ax.set_title(f'Class: {cal_class} Goal: {goal}')
        ax.set_xticks(np.arange(len(data.columns)))
        ax.set_yticks(np.arange(len(data.index)))
        ax.set_xticklabels(data.columns)
        ax.set_yticklabels(data.index)
        ax.spines[:].set_color('black')
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("S2", rotation=90)
        if show_plot:
            plt.show()
        return ax

    @staticmethod
    def heatmaps(result, **kwargs):
        """
        Plot S2 sensitivity results as a heatmap for multiple
        calibration classes and goals in one figure.

        :param pd.DataFrame result:
            A second order result from run
        :keyword [str] cal_class:
            Default is a list of all calibration classes in the result.
            If a list of classes is given only these classes are plotted.
        :keyword [str] goal:
            Default is a list of all goals in the result.
            If a list of goals is given only these goals are plotted.
        :keyword bool show_plot:
            Default is True. If False, all created plots are not shown.
        """
        show_plot = kwargs.pop('show_plot', True)
        cal_classes = kwargs.pop('cal_class', None)
        goals = kwargs.pop('goals', None)
        if cal_classes is None:
            cal_classes = result.index.get_level_values("Class").unique()
        if goals is None:
            goals = result.index.get_level_values("Goal").unique()

        fig, axes = plt.subplots(ncols=len(cal_classes), nrows=len(goals), sharex='all', sharey='all')
        if len(goals) == 1:
            axes = [axes]
        if len(cal_classes) == 1:
            for idx, ax in enumerate(axes):
                axes[idx] = [ax]

        for col, class_name in enumerate(cal_classes):
            for row, goal_name in enumerate(goals):
                SobolAnalyzer.heatmap(result, class_name, goal_name, ax=axes[row][col], show_plot=False)
        if show_plot:
            plt.show()

    def plot(self, result):
        """
        Plot the results of the sensitivity analysis method from run().

        :param pd.DataFrame result:
            Dataframe of the results like from the run() function.
        :return tuple of matplotlib objects (fig, ax)
        """
        SobolAnalyzer.plot_single(result=result[0])
        SobolAnalyzer.heatmaps(result=result[1])

    @staticmethod
    def load_second_order_from_csv(path):
        """
        Load second order sensitivity results which were saved with the run() or run_time_dependent() function.
        """
        result = pd.read_csv(path, index_col=[0, 1, 2, 3])
        return result
