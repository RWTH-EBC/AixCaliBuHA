
## Example 3 verbose sensitivity analysis
for the analysis of your model and the calibration process

Goals of this part of the examples:
1. Learn how to execute a verbose sensitivity analysis
2. Learn the meaning of the results and the analysis of your model
3  Learn how to use the results to select tuner-parameters for a calibration
4. Learn other sensitivity methods and compare them
5. Learn how to execute a time dependent sensitivity analysis
6. Learn how to save the results for reproduction

Start by importing all relevant packages

```python
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
from aixcalibuha import SobolAnalyzer, FASTAnalyzer, MorrisAnalyzer
from aixcalibuha.data_types import merge_calibration_classes
from examples import setup_fmu, setup_calibration_classes
from aixcalibuha import plotting
```

Please define the missing TODOs in the section below according to the docstrings.

```python
"""
Example process of a verbose sensitivity analysis for calibration and analysis porpoises.
First, the sensitivity problem is constructed, in this example
the `sobol` method is chosen.
Afterward, the SenAnalyzer class is instantiated to run the
sensitivity analysis in the next step.
The result of this analysis is then printed to the user.
and tuner-parameters are selected with these results.
A comparison between different methods is shown.
At the end the option to save a reproduction archive is shown.

:param [Path, str] examples_dir:
    Path to the examples folder of AixCaliBuHA
    e.g. r"LOCAL_PATH_TO\AixCaliBuHA\examples"
:param str example:
    Which example to run, "A" or "B"
:param int n_cpu:
    Number of cores to use

:return: A list of calibration classes
:rtype: list
"""
examples_dir = "TODO: Add a valid input according to the docstring above"
example: str  =  "B"
n_cpu: int  =  1
```

## Setup
Set up the class according to the documentation.
You just have to pass a valid simulation api and
some further settings for the analysis.
Let's first load the necessary simulation api:

```python
examples_dir = Path(examples_dir)
sim_api = setup_fmu(examples_dir=examples_dir, example=example, n_cpu=n_cpu)
```

To conduct a sensitivity analysis, we need to define calibration classes that
encompass the objectives (goals) for which sensitivity is to be assessed. In this
context, we'll explore distinct calibration classes corresponding to various
states of the models. Initially, we establish a comprehensive calibration class
that spans the entire duration, denoted as `global`. Subsequently, we can
leverage simulations from this global class for other classes targeting
specific time intervals within the overall global range.

For the specific states of the models with distinct time intervals, we adopt the
calibration classes from the second example and tailor them for the verbose
sensitivity analysis. Alternatively, custom classes could be created directly
function (`validation_class`), we opt to ignore it by using the variable
assignment `_` and omit any associated text output of the second example.

```python
calibration_classes, _ = setup_calibration_classes(
    examples_dir=examples_dir, example=example, multiple_classes=False
)
merged_calibration_classes = merge_calibration_classes(calibration_classes)
merged_calibration_classes[0].name = 'global'
```

Now, we add the calibration classes for the different states of the system

```python
calibration_classes, _ = setup_calibration_classes(
    examples_dir=examples_dir, example=example, multiple_classes=True
)
merged_calibration_classes.extend(merge_calibration_classes(calibration_classes))
```

This results in the following calibration classes where we merge the time intervals directly.
We could have also merged them with an option in the `run()` function of the
sensitivity analyzer classes.

```python
print("Calibration classes for sensitivity analysis:",
      [c.name for c in merged_calibration_classes])
```

In our detailed sensitivity analysis, it is essential for all classes to share
identical tuner parameters. This ensures that we can employ the same set of
simulations for calculating sensitivity across various calibration classes.
However, the final class in example B deviates in tuner parameters; hence, we
reset them to align with the tuner parameters of the other classes.

```python
if example == 'B':
    merged_calibration_classes[-1].tuner_paras = merged_calibration_classes[0].tuner_paras
```

With the definition of the calibration classes and the loaded `sim_api`,
we now take a look at the different options for sensitivity analysis.
First, we perform the `Sobol` method, which is the most powerful currently
supported method, but is also the most computational demanding one.
Afterward, we will compare the results of the different methods.

## Example of Sobol method
First, we instantiate the Analyzer with the `sim_api` and the number of samples.
For this example we will use a small sample size to reduce the time needed,
which will lead to inaccurate results.
In the comparison of the different methods, we will discuss the required sample size.
We will also define a working directory were all results of the analysis will be stored.
Additionally, we can choose if the samples and corresponding simulation files will be saved.
These files can later be loaded and used for analysis of different calibration classes
without performing new simulations. The simulations during the sensitivity analysis are the
main computational time factor.

```python
sen_analyzer = SobolAnalyzer(
    sim_api=sim_api,
    num_samples=8,
    calc_second_order=True,
    cd=examples_dir.joinpath('testzone', f'verbose_sen_{example}'),
    save_files=True,
    savepath_sim=examples_dir.joinpath('testzone', f'verbose_sen_{example}', 'files'),
    suffix_files='csv'
)
```

Now, we run the sensitivity analysis with the verbose option.
With that, we not only get the results for combined target values,
but we also get the results for every target value alone.
Because we defined the first calibration class global
and every calibration class has the same
tuner-parameters, we can use the option `use_first_sim`,
where only the first class will be simulated and the simulation files saved.
These simulations will be used for all other classes.

For that the simulations are loaded for each class
and the statistical measure is then evaluated for the
relevant time intervals of each class.

When we load simulation files we can use multiprocessing for loading the simulation
and evaluating the statistical measure for each class in their time intervals.
This multiprocessing option is especially
useful for large models and large simulation data,
because only one simulation at a time is stored
in memory for each process.
This can prevent possible memory errors.

We disable the automatic plot option here,
but we save all results. Later we can use the plot function
of the plotting module to plot the results.

```python
result, classes = sen_analyzer.run(calibration_classes=merged_calibration_classes,
                                   verbose=True,
                                   use_first_sim=True,
                                   plot_result=False,
                                   save_results=True,
                                   n_cpu=n_cpu,
                                   scale=False)
```

After running the sensitivity analysis you can see
that the working directory was created and the result
files were saved here. First, the folder "files" was
created as the result of `save_files=True`. In there, the
simulation files are stored in an own folder which
name contains the name of the used calibration class.
Additionally, the corresponding samples are stored.
The simulations are coupled with their samples.
As one results of the analysis, the statistical
measure and corresponding sample are saved for each class.
This information could be used for surrogate-based
calibration, which is currently not implemented in `AixCaliBuHA`.
The main results of the sensitivity analysis are the sensitivity
measures stored in "SobolAnalyzer_results.csv"
and "SobolAnalyzer_results_second_order.csv".
dataframes. This is specific to the sobol method, all other
of the sobol method with possibly other analysis variables.
Let´s take a look at these results.

The first result has as columns the tuner-parameters
and a multi level index with three levels.
The first level defines the calibration class.
The second level defines the Goals (target values). The index `all`
is for the result of the combined target values in the goals.
The last level defines the result of the
sensitivity measure for each class and goal.
These analysis variables are specific for each method.
For their exact meaning I refer to the documentation of the SALib or the literature.
In this example you get a short overview in the comparison later.

```python
print("Result of the sensitivity analysis")
print('First and total order results of sobol method')
print(result[0].to_string())
```

The second result of the sobol method is for second order sensitive measures.
These describe the interaction between two parameters,
so this dataframe has a fourth index level "Interaction".
In this level, the tuner-parameters are listed again.

```python
print('Second order results of sobol method')
print(result[1].to_string())
```

For a better understanding of the results we will now plot them.

## Plotting Sensitivity results
We start with the result which were calculated with the small sample size.
Let's plot the first and total order results. These results
are specific for each single parameter
For each calibration class, a figure is created
which shows for each goal the first order sensitivity `S1`
and the total order sensitivity `ST` combined. For the small
sample size the results have large confidence
intervals, which show that these results are inaccurate as we
noted earlier due to the small sample size.

```python
plotting.plot_single(result[0])
```

The plotting of second order results is only useful
and working for more than 2 parameters.
So we only can take a look at them in example A.
If you run example B we will skip the plot of the second order results
and load some sensitivity results of example A.

Let's take a look at the second order results `S2` of example A we just created.
This analysis variable shows the interaction of two
parameters, so we can plot them as a heatmap.
We can see that the parameters have no interaction with
themselves what is obvious. Also, we see that the
values for p1,p2 and p2,p1 are the same.
In the heatmap we can't visualize the confidence intervals,
so we will also take a look at the interaction of
one specific parameter.
For that the `plotting` module has also a function
which looks simular to the `S1` and `ST` plots.
Here we see again large confidence intervals,
so now we will load results which were calculated with
a much higher sample number.

```python
if example == 'A':
    plotting.heatmaps(result[1])
    plotting.plot_single_second_order(result[1], 'rad.n')
```

## Loading results
We will now load sensitivity results of example A .
These results were produced with this example and
a samples number N=1024 and calc_second_order=True

```python
result_sobol = SobolAnalyzer.load_from_csv(
    examples_dir.joinpath('data', 'SobolAnalyzer_results_A.csv')
)
result_sobol_2 = SobolAnalyzer.load_second_order_from_csv(
    examples_dir.joinpath('data', 'SobolAnalyzer_results_second_order_A.csv')
)
```

For a better understanding we will only take a
look at the global class and Electricity goal
and plot `S1`, `ST` and `S2` in the same window.
For that we can use the plot function
with some optional options. This shows how you can easily customize
these plots, and you can also chang everything
on the axes of the plots.

```python
fig = plt.figure(figsize=plt.figaspect(1. / 4.), layout="constrained")  # creating one figure
subfigs = fig.subfigures(1, 3, wspace=0)  # creating subfigures for each type of plot
ax0 = subfigs[0].subplots()  # plotting `S1` and `ST`
plotting.plot_single(
    result=result_sobol,
    cal_classes=['global'],
    goals=['Electricity'],
    show_plot=False,
    figs_axes=([subfigs[0]], [ax0]),
    max_name_len=14
)
ax1 = subfigs[1].subplots()  # plotting heatmap
plotting.heatmap(
    result_sobol_2,
    cal_class='global',
    goal='Electricity',
    ax=ax1,
    show_plot=False,
    max_name_len=14
)
ax2 = subfigs[2].subplots()  # plotting the interactions of one single parameter
plotting.plot_single_second_order(
    result=result_sobol_2,
    para_name='rad.n',
    show_plot=False,
    cal_classes=['global'],
    goals=['Electricity'],
    figs_axes=([subfigs[2]], [ax2]),
    max_name_len=14
)
plt.show()
```

Now, what can we see in these results? First, the
confidence intervals are now much smaller, so that we
can interpret and connect the different analysis variables.
`S1` stands for the variance in the objective,
which is caused by the variation of one parameter
while all other parameters are constant. The sobol analysis
variables are normalized with the total variance caused
by all parameter variations together within
their bounds. This means that when the parameters had
no interactions the sum of all `S1` values would
be 1. `ST` shows the resulting variance of a parameter
with all his interactions. Let's take a look at the
interaction between `n` and `G`, which are the highest.
Here, the `S2` value of the interaction
between `n` and `G` has a similar value to each difference
of `S1` and `ST` from these parameters. This show how this
difference corresponds to the interactions of a parameter.
All other parameters have only a very small sensitivity.
These are just some basics, to understand what option you
have in `AixCaliBuAH`. For more information look up relevant literature.

## Comparison of different Methods
We will now take a short look at a comparison of the
`Sobol`, `Fast` and `Morris` method in `AixCaliBuAH`.
The `SALib` provides more methods, which can
be implemented here in the future or on your own.
We already took a look at the `Sobol` method,
which can compute `S1`, `ST`, and `S2` with their confidence intervals.
The sobol method needs for that `(2+2k)N` simulations
where `k` is the number of parameters and `N` is the sample
number. For variance-based methods `N` should be greater
than 1000. The sobol method can also compute only `S1` and
`ST` with calc_second_order=False and (1+k)N simulations.
The FAST method is another variance-based
method and only computes `S1` and `ST` with k*N simulations.
`Sobol` and FAST should show simular results which is the
case for example B but in example A the FAST method overestimates `ST` in some cases.
In the right plots, the results for the `Morris` method are shown.
These are based on the mean of derivatives which
represents the analysis variables mu. In the estimation
of the derivatives, only one parameter is changed at a time.
mu_star is the mean of the absolut values of the derivatives
and is an approximation of `ST` but
needs only an `N` of over 100 and `(1+k)N` simulations.
In this comparison, mu_star shows simular results to `ST`
of the `Sobol` method, but it is not normalized.
Last, sigma is computed which is the standard deviation and
is a sign for a non-linear model or for interactions in the model.

```python
result_sobol = SobolAnalyzer.load_from_csv(
    examples_dir.joinpath('data', f'SobolAnalyzer_results_{example}.csv')
)
result_fast = FASTAnalyzer.load_from_csv(
    examples_dir.joinpath('data', f'FASTAnalyzer_results_{example}.csv')
)
result_morris = MorrisAnalyzer.load_from_csv(
    examples_dir.joinpath('data', f'MorrisAnalyzer_results_{example}.csv')
)

global_class = classes[0]
fig_comp = plt.figure(figsize=plt.figaspect(1. / 4.), layout="constrained")
subfigs_comp = fig_comp.subfigures(1, 3, wspace=0)
ax0_comp = subfigs_comp[0].subplots(3, 1, sharex=True)
plotting.plot_single(
    result=result_sobol,
    cal_classes=[global_class.name],
    show_plot=False,
    figs_axes=([subfigs_comp[0]], [ax0_comp])
)
subfigs_comp[0].suptitle("Sobol")
ax1_comp = subfigs_comp[1].subplots(3, 1, sharex=True)
plotting.plot_single(
    result=result_fast,
    cal_classes=[global_class.name],
    show_plot=False,
    figs_axes=([subfigs_comp[1]], [ax1_comp])
)
subfigs_comp[1].suptitle("FAST")
ax2_comp = subfigs_comp[2].subplots(3, 1, sharex=True)
plotting.plot_single(
    result=result_morris,
    show_plot=False,
    cal_classes=[global_class.name],
    figs_axes=([subfigs_comp[2]], [ax2_comp])
)
subfigs_comp[2].suptitle("Morris")
plt.show()
```

## Selection of tuner-parameters based on verbose sensitivity results
We can now also use these verbose sensitivity
results for a selection of relevant tuner parameters.
We already saw that our models have interactions,
so it will be necessary to calibrate them together
in one calibration class. The calibration class
global can be used for that because it includes all
the other specific classes. But in the sensitivity
results of this class it could be that parameters,
which are only in one state sensitive can be missed.
We can use the verbose sensitivity results so
that a parameter will be selected when it has a
sensitivity at least in one class and target value
of the sensitivity results. This is enough that
the parameter can be calibrated.
Here, we will use `S1` because it is normalized instead of mu_star,
and we can set on single threshold for all classes and goals.
Also, if a parameter is only through interaction sensitive (only in `ST`)
A real value can't be found and the parameters which interact will
compensate each others deviation to a real world value.
This still can happen when we choose with `S1` but it is less likely.

```python
calibration_class = SobolAnalyzer.select_by_threshold_verbose(classes[0],
                                                              result=result_sobol,
                                                              analysis_variable='S1',
                                                              threshold=0.001,
                                                              )

print(calibration_class.tuner_paras)
```

## Time dependent sensitivity
For analysis porpoises we can also evaluate the time dependent sensitivity
instead of the sensitivity of larger time intervals. But these results can not yet
be used to automatically selected tuner parameters for a calibration.
The function for the time dependent sensitivity is similar to the other one.
So we can also use the simulations of booth run functions for each other.
The main difference is that we only need one calibration class from which
the measured target data is not used and the sensitivity is directly
calculated for the change of the separate target values and no combined goals.
In the results, we then get just the additional index time.

```python
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # ignoring warnings that are caused by the low sample size
    result = sen_analyzer.run_time_dependent(
        cal_class=merged_calibration_classes[0],
        load_sim_files=True,
        plot_result=True
    )
print(result)
```

When we use the plot function we can see the sensitivity of each parameter
changing over time. We can also see that the confidence intervals
are again large for such a small sample size. When the confidence interval is larger
than one an info is shown and the confidence interval of the previous
time step is used to smooth it out for the visualisation.
Let's load again results which were created with a larger sample size.

## Loading time dependent results
These results were produced with a samples number `N=1024` and `calc_second_order=True`

```python
result_sobol_time = SobolAnalyzer.load_from_csv(
    examples_dir.joinpath('data', 'SobolAnalyzer_results_time_A.csv')
)
result_sobol_2_time = SobolAnalyzer.load_second_order_from_csv(
    examples_dir.joinpath('data', 'SobolAnalyzer_results_second_order_time_A.csv')
)
plotting.plot_time_dependent(result=result_sobol_time, plot_conf=True, show_plot=False)
```

Now the confidence intervals are smaller and
only at one time step they are still lager than 1.
We can also see that the parameter `theCon.G` has the biggest influence.
So we can take a closer look at this parameter with another plot
function where all available sensitivity measures are plotted together.
Second order results are plotted cumulative on top of `S1`.
This resembles the definition of `ST = S1 + sum(S2_i) + sum(S3_i) + ...`

```python
plotting.plot_parameter_verbose(parameter='theCon.G',
                                single_result=result_sobol_time,
                                second_order_result=result_sobol_2_time)
```

At the end we also can create a reproduction
archive which saves all settings and all created files
automatically with the reproduction function of ebcpy.
Not running in jupyter notebook

```python
file = sen_analyzer.save_for_reproduction(
    title="SenAnalyzerTest",
    path=examples_dir.joinpath('testzone'),
    log_message="This is just an example",
    remove_saved_files=False,
    exclude_sim_files=True
)
print("ZIP-File to reproduce all this:", file)
```
