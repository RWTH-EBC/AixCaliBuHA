"""
Module containing functions for plotting sensitivity results
"""
import warnings
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from SALib.plotting.bar import plot as barplot


def plot_single(result: pd.DataFrame,
                cal_classes: [str] = None,
                goals: [str] = None,
                show_plot: bool = True,
                use_suffix: bool = False,
                figs_axes: [[matplotlib.figure.Figure], [matplotlib.axes.Axes]] = None):
    """
    Plot sensitivity results of first and total order analysis variables.
    For each calibration class one figure is created, which shows for each goal an axis
    with a barplot of the values of the analysis variables.

    :param pd.DataFrame result:
        A result from run
    :param bool show_plot:
        Default is True. If False, all created plots are not shown.
    :param bool use_suffix:
        Default is True: If True, the last part after the last point
        of Modelica variables is used for the x ticks.
    :param [str] cal_classes:
        Default are all possible calibration classes. If a list of
        names of calibration classes is given only plots for these
        classes are created.
    :param [str] goals:
        Default are all possible goal names. If a list of specific
        goal names is given only these will be plotted.
    :param ([fig], [ax]) figs_axes:
        Default None. Useful for using subfigures (see example for verbose sensitivity analysis).
    :return:
        Returns all created figures and axes in lists like [fig], [ax]
    """

    # get lists of the calibration classes and their goals in the result dataframe
    if cal_classes is None:
        cal_classes = _del_duplicates(list(result.index.get_level_values(0)))
    if goals is None:
        goals = _del_duplicates(list(result.index.get_level_values(1)))

    # rename tuner_names in result to the suffix of their variable name
    if use_suffix:
        result = _rename_tuner_names(result)

    # when the index is not sorted pandas throws a performance warning
    result = result.sort_index()

    # plotting with simple plot function of the SALib
    figs = []
    axes = []
    for col, cal_class in enumerate(cal_classes):
        if figs_axes is None:
            fig, ax = plt.subplots(len(goals), sharex='all')
        else:
            fig = figs_axes[0][col]
            ax = figs_axes[1][col]
        fig.suptitle(cal_class)
        figs.append(fig)
        if not isinstance(ax, np.ndarray):
            ax = [ax]
        axes.append(ax)
        for row, goal in enumerate(goals):
            result_df = result.loc[cal_class, goal]
            axes[col][row].grid(True, which='both', axis='y')
            barplot(result_df.T, ax=axes[col][row])
            axes[col][row].set_title(goal)
            axes[col][row].legend()

    if show_plot:
        plt.show()

    return figs, axes


def plot_second_order(result: pd.DataFrame,
                      cal_classes: [str] = None,
                      goals: [str] = None,
                      show_plot: bool = True,
                      use_suffix: bool = False,
                      figs: [[matplotlib.figure.Figure]] = None):
    """
    Plot sensitivity results of second order analysis variables.
    For each calibration class and goal one figure of a 3d plot is created
    with the barplots of the interactions for each parameter.
    Only working for more than 2 parameter.

    :param pd.DataFrame result:
        A result from run
    :param bool show_plot:
        Default is True. If False, all created plots are not shown.
    :param bool use_suffix:
        Default is True: If True, the last part after the last point
        of Modelica variables is used for the x ticks.
    :param [str] cal_classes:
        Default are all possible calibration classes. If a list of
        names of calibration classes is given only plots for these
        classes are created.
    :param [str] goals:
        Default are all possible goal names. If a list of specific
        goal names is given only these will be plotted.
    :param [[fig]] figs:
        Default None. Useful for using subfigures (see example for verbose sensitivity analysis).
    :return:
        Returns all created figures and axes in lists like [fig], [ax]
    """
    result = result.fillna(0)
    if cal_classes is None:
        cal_classes = _del_duplicates(list(result.index.get_level_values(0)))
    if goals is None:
        goals = _del_duplicates(list(result.index.get_level_values(1)))

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


def plot_single_second_order(result: pd.DataFrame,
                             para_name: str,
                             cal_classes: [str] = None,
                             goals: [str] = None,
                             show_plot: bool = True,
                             use_suffix: bool = False,
                             figs_axes: [[matplotlib.figure.Figure], [matplotlib.axes.Axes]] = None):
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
    :param bool show_plot:
        Default is True. If False, all created plots are not shown.
    :param bool use_suffix:
        Default is True: If True, the last part after the last point
        of Modelica variables is used for the x ticks.
    :param ([fig], [ax]) figs_axes:
        Default None. Useful for using subfigures (see example for verbose sensitivity analysis).
    :return:
        Returns all created figures and axes in lists like [fig], [ax]
    """
    result = result.loc[:, :, :, para_name][:].fillna(0)
    figs, axes = plot_single(
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


def heatmap(result: pd.DataFrame,
            cal_class: str,
            goal: str,
            ax: matplotlib.axes.Axes = None,
            show_plot: bool = True,
            use_suffix: bool = False):
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


def heatmaps(result: pd.DataFrame,
             cal_classes: [str] = None,
             goals: [str] = None,
             show_plot: bool = True,
             use_suffix: bool = False):
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
    :param bool show_plot:
        Default is True. If False, all created plots are not shown.
    :param bool use_suffix:
        Default is False. If True, only the last suffix of a Modelica variable is displayed.
    """
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
            heatmap(result, class_name, goal_name, ax=axes[row][col], show_plot=False, use_suffix=use_suffix)
    if show_plot:
        plt.show()


def plot_time_dependent(result: pd.DataFrame,
                        parameters: [str] = None,
                        goals: [str] = None,
                        analysis_variables: [str] = None,
                        plot_conf: bool = True,
                        show_plot: bool = True,
                        use_suffix: bool = False,
                        figs_axes: [[matplotlib.figure.Figure], [matplotlib.axes.Axes]] = None):
    """
    Plot time dependent sensitivity results without interactions from run_time_dependent().

    For each goal one figure is created with one axes for each analysis variable. In these plots the time dependent sensitivity of the parameters is plotted.
    The confidence interval can also be plotted.

    :param pd.DataFrame result:
        A result from run_time_dependent without second order results.
    :param [str] parameters:
        Default all parameters. List of parameters to plot the sensitivity.
    :param [str] analysis_variables:
        Default all analysis_variables. List of analysis variables to plot.
    :param bool plot_conf:
        Default True. If true, the confidence intervals for each parameter are plotted.
    :param bool show_plot:
        Default is True. If False, all created plots are not shown.
    :param bool use_suffix:
        Default is True: If True, the last part after the last point
        of Modelica variables is used for the x ticks.
    :param [str] goals:
        Default are all possible goal names. If a list of specific
        goal names is given only these will be plotted.
    :param ([fig], [ax]) figs_axes:
        Default None. Optional custom figures and axes (see example for verbose sensitivity analysis).
    :return:
        Returns all created figures and axes in lists like [fig], [ax]
    """
    if goals is None:
        goals = _del_duplicates(list(result.index.get_level_values(0)))
    all_analysis_variables = _del_duplicates(list(result.index.get_level_values(1)))
    if analysis_variables is None:
        analysis_variables = [av for av in all_analysis_variables if '_conf' not in av]
    if parameters is None:
        parameters = result.columns.values

    # rename tuner_names in result to the suffix of their variable name
    if use_suffix:
        result = _rename_tuner_names(result)

    # when the index is not sorted pandas throws a performance warning
    result = result.sort_index()

    figs = []
    axes = []
    for g_i, goal in enumerate(goals):
        if figs_axes is None:
            fig, ax = plt.subplots(len(analysis_variables), sharex='all')
        else:
            fig = figs_axes[0][g_i]
            ax = figs_axes[1][g_i]
        fig.suptitle(goal)
        figs.append(fig)
        if not isinstance(ax, np.ndarray):
            ax = [ax]
        axes.append(ax)
        for av_i, av in enumerate(analysis_variables):
            axes[g_i][av_i].plot(result.loc[goal, av][parameters])
            axes[g_i][av_i].set_ylabel(av)
            axes[g_i][av_i].legend(parameters)
            if plot_conf and av + '_conf' in all_analysis_variables:
                for p in parameters:
                    y = result.loc[goal, av][p]
                    x = y.index.to_numpy()
                    ci = result.loc[goal, av + '_conf'][p]
                    large_values_indices = ci[ci > 1].index
                    if list(large_values_indices):
                        warnings.warn(
                            f"Confidence interval for {goal}, {av}, {p} was at the "
                            f"following times {list(large_values_indices)} lager than 1 "
                            f"and is smoothed out in the plot.")
                    for idx in large_values_indices:
                        prev_idx = ci.index.get_loc(idx) - 1
                        if prev_idx >= 0:
                            ci.iloc[ci.index.get_loc(idx)] = ci.iloc[prev_idx]
                        else:
                            ci.iloc[ci.index.get_loc(idx)] = 1
                    axes[g_i][av_i].fill_between(x, (y - ci), (y + ci), alpha=.1)
        axes[g_i][-1].set_xlabel('time')
    if show_plot:
        plt.show()
    return figs, axes


def plot_parameter_verbose(parameter: str,
                           single_result: pd.DataFrame,
                           second_order_result: pd.DataFrame = None,
                           goals: [str] = None,
                           show_plot: bool = True,
                           use_suffix: bool = False,
                           fig_axes: [matplotlib.figure.Figure, [matplotlib.axes.Axes]] = None):
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
    :param bool show_plot:
        Default is True. If False, all created plots are not shown.
    :param bool use_suffix:
        Default is True: If True, the last part after the last point
        of Modelica variables is used for the x ticks.
    :param [str] goals:
        Default are all possible goal names. If a list of specific
        goal names is given only these will be plotted.
    :param (fig, [ax]) fig_axes:
        Default None. Optional custom figures and axes (see example for verbose sensitivity analysis).
    :return:
        Returns all created figures and axes in lists like [fig], [ax]
    """
    if goals is None:
        goals = _del_duplicates(list(single_result.index.get_level_values(0)))
    all_analysis_variables = _del_duplicates(list(single_result.index.get_level_values(1)))
    analysis_variables = [av for av in all_analysis_variables if '_conf' not in av]

    # rename tuner_names in result to the suffix of their variable name
    if use_suffix:
        single_result = _rename_tuner_names(single_result)
        # when the index is not sorted pandas throws a performance warning
        single_result = single_result.sort_index()
        if second_order_result is not None:
            second_order_result = _rename_tuner_names(second_order_result)
            second_order_result = second_order_result.sort_index()

    if fig_axes is None:
        fig, ax = plt.subplots(len(goals), sharex='all')
    else:
        fig = fig_axes[0]
        ax = fig_axes[1]
    fig.suptitle(parameter)
    if not isinstance(ax, np.ndarray):
        ax = [ax]
    for g_i, goal in enumerate(goals):
        if second_order_result is not None:
            result_2_goal = second_order_result.loc[goal, 'S2', parameter]
            mean = result_2_goal.mean().drop([parameter])
            mean.sort_values(ascending=False, inplace=True)
            sorted_interactions = list(mean.index)
            time_ar = _del_duplicates(list(result_2_goal.index.get_level_values(0)))
            value = single_result.loc[goal, 'S1'][parameter].to_numpy()
            ax[g_i].plot(single_result.loc[goal, 'S1'][parameter], label='S1')
            ax[g_i].fill_between(time_ar, np.zeros_like(value), value, alpha=0.1)
            for para in sorted_interactions:
                value_2 = value + result_2_goal[para].to_numpy()
                ax[g_i].plot(time_ar, value_2, label='S2 ' + para)
                ax[g_i].fill_between(time_ar, value, value_2, alpha=0.1)
                value = value_2
            ax[g_i].plot(single_result.loc[goal, 'ST'][parameter], label='ST')
            legend = ['S1']
            legend.extend(analysis_variables)
            legend.append('ST')
            ax[g_i].set_title(goal)
            ax[g_i].legend()
        else:
            for av_i, av in enumerate(analysis_variables):
                ax[g_i].plot(single_result.loc[goal, av][parameter])
            ax[g_i].legend(analysis_variables)
    if show_plot:
        plt.show()
    return fig, ax


def _del_duplicates(x):
    """Helper function"""
    return list(dict.fromkeys(x))


def _rename_tuner_names(result):
    """Helper function"""
    tuner_names = list(result.columns)
    rename_tuner_names = {name: _get_suffix(name) for name in tuner_names}
    result = result.rename(columns=rename_tuner_names, index=rename_tuner_names)
    return result


def _get_suffix(modelica_var_name):
    """Helper function"""
    index_last_dot = modelica_var_name.rfind('.')
    suffix = modelica_var_name[index_last_dot + 1:]
    return suffix
