 # Example 3 sensitivity analysis
 Goals of this part of the examples:
 1. Learn how to execute a sensitivity analysis
 2. Learn how to automatically select sensitive tuner parameters

 Import a valid analyzer, e.g. `SobolAnalyzer`
```python
from aixcalibuha import SobolAnalyzer
```
 Please define the missing TODOs in the section below according to the docstrings.
```python
"""
Example process of a sensitivity analysis.
First, the sensitivity problem is constructed, in this example
the `morris` method is chosen.
Afterwards, the sen_analyzer class is instantiated to run the
sensitivity analysis in the next step.
The result of this analysis is then printed to the user.
The automatic_select function is presented as-well, using a threshold of 1
and the default `mu_star` criterion.

:param [pathlib.Path, str] examples_dir:
    Path to the examples folder of AixCaliBuHA
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
 Setup the class according to the documentation.
 You just have to pass a valid simulation api and
 some further settings for the analysis.
 Let's thus first load the necessary simulation api:
```python
from examples import setup_fmu, setup_calibration_classes
sim_api = setup_fmu(examples_dir=examples_dir, example=example, n_cpu=n_cpu)

sen_analyzer = SobolAnalyzer(
        sim_api=sim_api,
        num_samples=10,
        cd=sim_api.cd,
        analysis_variable='S1'
    )
```
 Now perform the analysis for the one of the given calibration classes.
```python
calibration_classes = setup_calibration_classes(
    examples_dir=examples_dir, example=example
)[0]

result, classes = sen_analyzer.run(calibration_classes=calibration_classes,
                                   plot_result=True,
                                   save_results=False)
print("Result of the sensitivity analysis")
print(result)
```
 For each given class, you should see the given tuner parameters
 and the sensitivity according to the selected method from the SALib.
 Let's remove some less sensitive parameters based on some threshold
 to remove complexity from our calibration problem:
```python
print("Selecting relevant tuner-parameters using a fixed threshold:")
sen_analyzer.select_by_threshold(calibration_classes=classes,
                                 result=result[0],
                                 threshold=0.01,
                                 analysis_variable='S1')
for cal_class in classes:
    print(f"Class '{cal_class.name}' with parameters:\n{cal_class.tuner_paras}")
```
 Return the classes and the sim_api to later perform an automated process in example 5
