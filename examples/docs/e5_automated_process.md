 # Example 5 Automated process
 Goals of this part of the examples:
 1. Learn how to run everything in one script

 Start by importing everything
```python
from examples import setup_fmu, setup_calibration_classes
from examples.e3_sensitivity_analysis_example import run_sensitivity_analysis
from examples.e4_calibration_example import run_calibration


"""
Arguments of this example:

:param str example:
    Whether to use example A (requires windows) or B.
    Default is "A"
"""
example = "A"
```
 First we run the sensitivity analysis:
```python
calibration_classes, sim_api = run_sensitivity_analysis(example=example)
```
 Then the calibration and validation
```python
run_calibration(example=example,
                sim_api=sim_api,
                cal_classes=calibration_classes)
```
