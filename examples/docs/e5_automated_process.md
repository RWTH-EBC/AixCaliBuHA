 # Example 5 Automated process
 Goals of this part of the examples:
 1. Learn how to run everything in one script

 Start by importing everything
```python
from examples.e3_sensitivity_analysis_example import run_sensitivity_analysis
from examples.e4_calibration_example import run_calibration
```
 First we run the sensitivity analysis:
```python
calibration_classes, sim_api = run_sensitivity_analysis(
    examples_dir=examples_dir, example=example, n_cpu=n_cpu
)
```
 Then the calibration and validation
```python
run_calibration(
    examples_dir=examples_dir,
    example=example,
    sim_api=sim_api,
    cal_classes=calibration_classes
)
```
