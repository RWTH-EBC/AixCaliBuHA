
# Example 5 Automated process

Goals of this part of the examples:
1. Learn how to run everything in one script

Start by importing everything

```python
from examples.e3_sensitivity_analysis_example import run_sensitivity_analysis
from examples.e4_calibration_example import run_calibration
```

Please define the missing TODOs in the section below according to the docstrings.

```python
"""
Arguments of this example:

:param [pathlib.Path, str] examples_dir:
    Path to the examples folder of AixCaliBuHA
:param str example:
    Whether to use example A (requires windows) or B.
    Default is "A"
:param int n_cpu:
    Number of cores to use
"""
examples_dir = "TODO: Add a valid input according to the docstring above"
example: str  =  "A"
n_cpu: int  =  1
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
