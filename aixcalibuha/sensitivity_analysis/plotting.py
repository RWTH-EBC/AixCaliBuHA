"""
Module containing functions for plotting sensitivity results
"""
import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from SALib.plotting.bar import plot as barplot
from aixcalibuha.utils.visualizer import short_name


def plot_single(result: pd.DataFrame,
                cal_classes: [str] = None,
                goals: [str] = None,
                max_name_len: int = 15,
                **kwargs):
    """
    Plot sensitivity results of first and total order analysis variables.
    For each calibration class one figure is created, which shows for each goal an axis
    with a barplot of the values of the analysis variables.

    :param pd.DataFrame result:
        A result from run
    :param int max_name_len:
        Default is 10. Shortens the parameter names to max_name_len characters.
    :param [str] cal_classes:
        Default are all possible calibration classes. If a list of
        names of calibration classes is given only plots for these
        classes are created.
    :param [str] goals:
        Default are all possible goal names. If a list of specific
        goal names is given only these will be plotted.
    :keyword bool show_plot:
        Default is True. If False, all created plots are not shown.
    :keyword ([fig], [ax]) figs_axes:
        Default None. Set own figures of subfigures with corresponding axes for customization.
    :keyword bool use_suffix:
        Default is True: If True, the last part after the last point
        of Modelica variables is used for the x ticks.
    :return:
        Returns all created figures and axes in lists like [fig], [ax]
        with shapes (len(cal_classes)), (len(cal_classes), len(goals))
    """
    use_suffix = kwargs.pop('use_suffix', False)
    show_plot = kwargs.pop('show_plot', True)
    figs_axes = kwargs.pop('figs_axes', None)
    _func_name = kwargs.pop('_func_name', 'plot_single')

    # get lists of the calibration classes and their goals in the result dataframe
    if cal_classes is None:
        cal_classes = _del_duplicates(list(result.index.get_level_values(0)))
    if goals is None:
        goals = _del_duplicates(list(result.index.get_level_values(1)))

    result = _rename_tuner_names(result, use_suffix, max_name_len, _func_name)

    # plotting with simple plot function of the SALib
    figs = []
    axes = []
    for col, cal_class in enumerate(cal_classes):
        fig, ax = _create_figs_axes(figs_axes, col, goals, f"Class: {cal_class}")
        figs.append(fig)
        axes.append(ax)
        for row, goal in enumerate(goals):
            result_df = result.loc[cal_class, goal]
            axes[col][row].grid(True, which='both', axis='y')
            barplot(result_df.T, ax=axes[col][row])
            axes[col][row].set_title(f"Goal: {goal}")
            axes[col][row].legend()

    if show_plot:
        plt.show()

    return figs, axes


def plot_second_order(result: pd.DataFrame,
                      cal_classes: [str] = None,
                      goals: [str] = None,
                      max_name_len: int = 15,
                      **kwargs):
    """
    Plot sensitivity results of second order analysis variables.
    For each calibration class and goal one figure of a 3d plot is created
    with the barplots of the interactions for each parameter.
    Only working for more than 2 parameter.

    :param pd.DataFrame result:
        A result from run
    :param int max_name_len:
        Default is 10. Shortens the parameter names to max_name_len characters.
    :param [str] cal_classes:
        Default are all possible calibration classes. If a list of
        names of calibration classes is given only plots for these
        classes are created.
    :param [str] goals:
        Default are all possible goal names. If a list of specific
        goal names is given only these will be plotted.
    :keyword bool show_plot:
        Default is True. If False, all created plots are not shown.
    :keyword bool use_suffix:
        Default is True: If True, the last part after the last point
        of Modelica variables is used for the x ticks.
    :keyword [[fig]] figs:
        Default None. Set own figures of subfigures for customization.
        Shape (len(cal_classes), len(goals))
    :return:
        Returns all created figures and axes in lists like [fig], [ax]
    """
    use_suffix = kwargs.pop('use_suffix', False)
    show_plot = kwargs.pop('show_plot', True)
    figs = kwargs.pop('figs', None)
    result = result.fillna(0)
    if cal_classes is None:
        cal_classes = _del_duplicates(list(result.index.get_level_values(0)))
    if goals is None:
        goals = _del_duplicates(list(result.index.get_level_values(1)))

    result = _rename_tuner_names(result, use_suffix, max_name_len, "plot_second_order")

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
                ax.bar(tuner_names,
                       result.loc[cal_class, goal, 'S2', name].to_numpy(),
                       zs=idx, zdir='y', alpha=0.8)
                ax.set_title(f"Class: {cal_class} Goal: {goal}")
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


def plot_single_second_order(result: pd.DataFrame,
                             para_name: str,
                             cal_classes: [str] = None,
                             goals: [str] = None,
                             max_name_len: int = 15,
                             **kwargs):
    """
    Plot the value of S2 from one parameter with all other parameters.

    :param pd.DataFrame result:
        Second order result from run.
    :param str para_name:
        Name of the parameter of which the results should be plotted.
    :param [str] cal_classes:
        Default are all possible calibration classes. If a list of
        names of calibration classes is given only plots for these
        classes are created.
    :param [str] goals:
        Default are all possible goal names. If a list of specific
        goal names is given only these will be plotted.
    :param int max_name_len:
        Default is 10. Shortens the parameter names to max_name_len characters.
    :keyword bool show_plot:
        Default is True. If False, all created plots are not shown.
    :keyword bool use_suffix:
        Default is True: If True, the last part after the last point
        of Modelica variables is used for the x ticks.
    :keyword ([fig], [ax]) figs_axes:
        Default None. Set own figures of subfigures with corresponding axes for customization.
    :return:
        Returns all created figures and axes in lists like [fig], [ax]
        with shapes (len(cal_classes)), (len(cal_classes), len(goals))
    """
    use_suffix = kwargs.pop('use_suffix', False)
    show_plot = kwargs.pop('show_plot', True)
    figs_axes = kwargs.pop('figs_axes', None)

    result = result.loc[:, :, :, para_name][:].fillna(0)
    figs, axes = plot_single(
        result=result,
        show_plot=False,
        cal_classes=cal_classes,
        goals=goals,
        figs_axes=figs_axes,
        use_suffix=use_suffix,
        max_name_len=max_name_len,
        _func_name="plot_single_second_order"
    )
    # set new title for the figures of each calibration class
    for fig in figs:
        fig.suptitle(f"Interaction of {para_name} in class {fig._suptitle.get_text()}")
    if show_plot:
        plt.show()
    return figs, axes


def heatmap(result: pd.DataFrame,
            cal_class: str,
            goal: str,
            ax: matplotlib.axes.Axes = None,
            max_name_len: int = 15,
            **kwargs):
    """
    Plot S2 sensitivity results from one calibration class and goal as a heatmap.

    :param pd.DataFrame result:
        A second order result from run
    :param str cal_class:
        Name of the class to plot S2 from.
    :param str goal:
        Name of the goal to plot S2 from.
    :param matplotlib.axes ax:
        Default is None. If an axes is given the heatmap will be plotted on it, else
        a new figure and axes is created.
    :param int max_name_len:
        Default is 10. Shortens the parameter names to max_name_len characters.
    :keyword bool show_plot:
        Default is True. If False, all created plots are not shown.
    :keyword bool use_suffix:
        Default is False. If True, only the last suffix of a Modelica variable is displayed.
    :return:
        Returns axes
    """
    use_suffix = kwargs.pop('use_suffix', False)
    show_plot = kwargs.pop('show_plot', True)
    _func_name = kwargs.pop('_func_name', "heatmap")

    result = _rename_tuner_names(result, use_suffix, max_name_len, _func_name)
    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    data = result.sort_index().loc[cal_class, goal, 'S2'].fillna(0).reindex(
        index=result.columns)
    image = ax.imshow(data, cmap='Reds')
    ax.set_title(f'Class: {cal_class} Goal: {goal}')
    ax.set_xticks(np.arange(len(data.columns)))
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_xticklabels(data.columns)
    ax.set_yticklabels(data.index)
    ax.spines[:].set_color('black')
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
    cbar = ax.figure.colorbar(image, ax=ax)
    cbar.ax.set_ylabel("S2", rotation=90)
    if show_plot:
        plt.show()
    return ax


def heatmaps(result: pd.DataFrame,
             cal_classes: [str] = None,
             goals: [str] = None,
             max_name_len: int = 15,
             **kwargs):
    """
    Plot S2 sensitivity results as a heatmap for multiple
    calibration classes and goals in one figure.

    :param pd.DataFrame result:
        A second order result from run
    :param [str] cal_classes:
        Default is a list of all calibration classes in the result.
        If a list of classes is given only these classes are plotted.
    :param [str] goals:
        Default is a list of all goals in the result.
        If a list of goals is given only these goals are plotted.
    :param int max_name_len:
        Default is 10. Shortens the parameter names to max_name_len characters.
    :keyword bool show_plot:
        Default is True. If False, all created plots are not shown.
    :keyword bool use_suffix:
        Default is False. If True, only the last suffix of a Modelica variable is displayed.
    """
    use_suffix = kwargs.pop('use_suffix', False)
    show_plot = kwargs.pop('show_plot', True)

    if cal_classes is None:
        cal_classes = result.index.get_level_values("Class").unique()
    if goals is None:
        goals = result.index.get_level_values("Goal").unique()

    _, axes = plt.subplots(ncols=len(cal_classes), nrows=len(goals), sharex='all', sharey='all',
                           layout="constrained")
    if len(goals) == 1:
        axes = [axes]
    if len(cal_classes) == 1:
        for idx, ax in enumerate(axes):
            axes[idx] = [ax]
    _func_name = "heatmaps"
    for col, class_name in enumerate(cal_classes):
        for row, goal_name in enumerate(goals):
            heatmap(result,
                    class_name,
                    goal_name,
                    ax=axes[row][col],
                    show_plot=False,
                    use_suffix=use_suffix,
                    max_name_len=max_name_len,
                    _func_name=_func_name)
            _func_name = None
    if show_plot:
        plt.show()


def plot_time_dependent(result: pd.DataFrame,
                        parameters: [str] = None,
                        goals: [str] = None,
                        analysis_variables: [str] = None,
                        plot_conf: bool = True,
                        **kwargs):
    """
    Plot time dependent sensitivity results without interactions from run_time_dependent().

    For each goal one figure is created with one axes for each analysis variable.
    In these plots the time dependent sensitivity of the parameters is plotted.
    The confidence interval can also be plotted.

    :param pd.DataFrame result:
        A result from run_time_dependent without second order results.
    :param [str] parameters:
        Default all parameters. List of parameters to plot the sensitivity.
    :param [str] analysis_variables:
        Default all analysis_variables. List of analysis variables to plot.
    :param bool plot_conf:
        Default True. If true, the confidence intervals for each parameter are plotted.
    :param [str] goals:
        Default are all possible goal names. If a list of specific
        goal names is given only these will be plotted.
    :keyword ([fig], [ax]) figs_axes:
        Default None. Optional custom figures and axes
        (see example for verbose sensitivity analysis).
    :return:
        Returns all created figures and axes in lists like [fig], [ax]
        with shapes (len(goals)), (len(goals), len(analysis_variables))
    :keyword bool show_plot:
        Default is True. If False, all created plots are not shown.
    :keyword bool use_suffix:
        Default is True: If True, the last part after the last point
        of Modelica variables is used for the x ticks.
    :keyword int max_name_len:
        Default is 50. Shortens the parameter names to max_name_len characters.
    """
    use_suffix = kwargs.pop('use_suffix', False)
    max_name_len = kwargs.pop('max_name_len', 50)
    show_plot = kwargs.pop('show_plot', True)
    figs_axes = kwargs.pop('figs_axes', None)

    if goals is None:
        goals = _del_duplicates(list(result.index.get_level_values(0)))
    all_analysis_variables = _del_duplicates(list(result.index.get_level_values(1)))
    if analysis_variables is None:
        analysis_variables = [av for av in all_analysis_variables if '_conf' not in av]
    if parameters is None:
        parameters = result.columns.values

    result = _rename_tuner_names(result, use_suffix, max_name_len, "plot_time_dependent")

    renamed_parameters = [_format_name(para, use_suffix, max_name_len) for para in parameters]

    figs = []
    axes = []
    for idx_goal, goal in enumerate(goals):
        fig, ax = _create_figs_axes(figs_axes, idx_goal, analysis_variables, f"Goal: {goal}")
        figs.append(fig)
        axes.append(ax)
        for idx_av, analysis_var in enumerate(analysis_variables):
            axes[idx_goal][idx_av].plot(result.loc[goal, analysis_var][renamed_parameters])
            axes[idx_goal][idx_av].set_ylabel(analysis_var)
            axes[idx_goal][idx_av].legend(renamed_parameters)
            if plot_conf and analysis_var + '_conf' in all_analysis_variables:
                for para in renamed_parameters:
                    y = result.loc[goal, analysis_var][para]
                    x = y.index.to_numpy()
                    conv_int = result.loc[goal, analysis_var + '_conf'][para]
                    large_values_indices = conv_int[conv_int > 1].index
                    if list(large_values_indices):
                        sys.stderr.write(
                            f"plot_time_dependent INFO:"
                            f"Confidence interval for {goal}, {analysis_var}, {para} was at the "
                            f"following times {list(large_values_indices)} lager than 1 "
                            f"and is smoothed out in the plot.\n")
                    for idx in large_values_indices:
                        prev_idx = conv_int.index.get_loc(idx) - 1
                        if prev_idx >= 0:
                            conv_int.iloc[conv_int.index.get_loc(idx)] = conv_int.iloc[prev_idx]
                        else:
                            conv_int.iloc[conv_int.index.get_loc(idx)] = 1
                    axes[idx_goal][idx_av].fill_between(x, (y - conv_int), (y + conv_int), alpha=.1)
        axes[idx_goal][-1].set_xlabel('time')
    if show_plot:
        plt.show()
    return figs, axes


def plot_parameter_verbose(parameter: str,
                           single_result: pd.DataFrame,
                           second_order_result: pd.DataFrame = None,
                           goals: [str] = None,
                           **kwargs):
    """
    Plot all time dependent sensitivity measure for one parameter.
    For each goal an axes is created within one figure.

    If second_order_results form SobolAnalyzer.run_time_dependent are given
    the S2 results of the interaction with each other parameter are added on top
    of each other and the first order result.

    :param str parameter:
        Parameter to plot all sensitivity results for. If use_suffix=True, then
        the name must also be only the suffix.
    :param pd.DataFrame single_result:
        First and total order result form run_time_dependent.
    :param pd.DataFrame second_order_result:
        Default None. Second order result of SobolAnalyzer.run_time_dependent.
    :param [str] goals:
        Default are all possible goal names. If a list of specific
        goal names is given only these will be plotted.
    :keyword (fig, [ax]) fig_axes:
        Default None. Optional custom figures and axes
        (see example for verbose sensitivity analysis).
    :return:
        Returns all created figures and axes in lists like fig, [ax]
        with shape (len(goals)) for the axes list
    :keyword bool show_plot:
        Default is True. If False, all created plots are not shown.
    :keyword bool use_suffix:
        Default is True: If True, the last part after the last point
        of Modelica variables is used for the x ticks.
    :keyword int max_name_len:
        Default is 10. Shortens the parameter names to max_name_len characters.
    """
    use_suffix = kwargs.pop('use_suffix', False)
    max_name_len = kwargs.pop('max_name_len', 50)
    show_plot = kwargs.pop('show_plot', True)
    fig_axes = kwargs.pop('fig_axes', None)

    if goals is None:
        goals = _del_duplicates(list(single_result.index.get_level_values(0)))
    all_analysis_variables = _del_duplicates(list(single_result.index.get_level_values(1)))
    analysis_variables = [av for av in all_analysis_variables if '_conf' not in av]

    renamed_parameter = _format_name(parameter, use_suffix, max_name_len)

    single_result = _rename_tuner_names(single_result, use_suffix, max_name_len,
                                        "plot_parameter_verbose")
    if second_order_result is not None:
        second_order_result = _rename_tuner_names(second_order_result, use_suffix, max_name_len)

    if fig_axes is None:
        fig, ax = plt.subplots(len(goals), sharex='all', layout="constrained")
    else:
        fig = fig_axes[0]
        ax = fig_axes[1]
    fig.suptitle(f"Parameter: {parameter}")
    if not isinstance(ax, np.ndarray):
        ax = [ax]
    for g_i, goal in enumerate(goals):
        if second_order_result is not None:
            result_2_goal = second_order_result.loc[goal, 'S2', renamed_parameter]
            mean = result_2_goal.mean().drop([renamed_parameter])
            mean.sort_values(ascending=False, inplace=True)
            sorted_interactions = list(mean.index)
            time_ar = _del_duplicates(list(result_2_goal.index.get_level_values(0)))
            value = single_result.loc[goal, 'S1'][renamed_parameter].to_numpy()
            ax[g_i].plot(single_result.loc[goal, 'S1'][renamed_parameter], label='S1')
            ax[g_i].fill_between(time_ar, np.zeros_like(value), value, alpha=0.1)
            for para in sorted_interactions:
                value_2 = value + result_2_goal[para].to_numpy()
                ax[g_i].plot(time_ar, value_2, label='S2 ' + para)
                ax[g_i].fill_between(time_ar, value, value_2, alpha=0.1)
                value = value_2
            ax[g_i].plot(single_result.loc[goal, 'ST'][renamed_parameter], label='ST')
            legend = ['S1']
            legend.extend(analysis_variables)
            legend.append('ST')
            ax[g_i].set_title(f"Goal: {goal}")
            ax[g_i].legend()
        else:
            for analysis_var in analysis_variables:
                ax[g_i].plot(single_result.loc[goal, analysis_var][renamed_parameter])
            ax[g_i].legend(analysis_variables)
    if show_plot:
        plt.show()
    return fig, ax


def _del_duplicates(x):
    """Helper function"""
    return list(dict.fromkeys(x))


def _rename_tuner_names(result, use_suffix, max_len, func_name=None):
    """Helper function"""
    tuner_names = list(result.columns)
    renamed_names = {name: _format_name(name, use_suffix, max_len) for name in tuner_names}
    result = result.rename(columns=renamed_names, index=renamed_names)
    result = result.sort_index()
    for old, new in renamed_names.items():
        if old != new and func_name is not None:
            sys.stderr.write(f"{func_name} INFO: parameter name {old} changed to {new}\n")
    return result


def _format_name(name, use_suffix, max_len):
    """
    Format tuner names.
    """
    if use_suffix:
        name = _get_suffix(name)
    name = short_name(name, max_len)
    return name


def _get_suffix(modelica_var_name):
    """Helper function"""
    index_last_dot = modelica_var_name.rfind('.')
    suffix = modelica_var_name[index_last_dot + 1:]
    return suffix


def _create_figs_axes(figs_axes, fig_index, ax_len_list, fig_title):
    """
    Check if figs and axes are already given, if not create them.
    """
    if figs_axes is None:
        fig, ax = plt.subplots(len(ax_len_list), sharex='all', layout="constrained")
    else:
        fig = figs_axes[0][fig_index]
        ax = figs_axes[1][fig_index]
    fig.suptitle(fig_title)
    if not isinstance(ax, np.ndarray):
        ax = [ax]
    return fig, ax
