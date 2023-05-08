 # Example 3 sensitivity analysis
 Goals of this part of the examples:
 1. Learn the settings for a calibration
 2. Learn how to use both Single- and MultiClassCalibration
 3. Learn how to validate your calibration
```python
import numpy as np
from aixcalibuha import CalibrationClass, Calibrator, MultipleClassCalibrator
```
 ## Setup
 Start by loading the simulation api and the calibration classes
```python
from examples import setup_fmu, setup_calibration_classes
if sim_api is None:
    sim_api = setup_fmu(examples_dir=examples_dir, example=example, n_cpu=n_cpu)
default_cal_classes, validation_class = setup_calibration_classes(
    examples_dir=examples_dir,
    example=example,
    multiple_classes=False
)
if cal_classes is None:
    cal_classes = default_cal_classes
```
 ## Calibration and optimization settings
 We refer to the docstrings on more information on each setting.
 Specify values for keyword-arguments to customize
 the Calibration process for a single-class calibration
```python
kwargs_calibrator = {"timedelta": 0,
                     "save_files": False,
                     "verbose_logging": True,
                     "show_plot": True,
                     "create_tsd_plot": True,
                     "save_tsd_plot": True,
                     "show_plot_pause_time": 1e-3,
                     "plot_file_type": "png",
                     "fail_on_error": False,
                     "ret_val_on_error": np.NAN,
                     # For this example, let's keep the runtime low
                     "max_itercount": 100
                     }
```
 Specify values for keyword-arguments to customize
 the Calibration process for a multiple-class calibration
```python
kwargs_multiple_classes = {"merge_multiple_classes": True}
```
 Specify solver-specific keyword-arguments depending on the solver and method you will use
```python
kwargs_scipy_dif_evo = {"maxiter": 30,
                        "popsize": 5,
                        "mutation": (0.5, 1),
                        "recombination": 0.7,
                        "seed": None,
                        "polish": True,
                        "init": 'latinhypercube',
                        "atol": 0}
kwargs_dlib_min = {"num_function_calls": int(1e9),
                   "solver_epsilon": 0}
kwargs_scipy_min = {"tol": None,
                    "options": {"maxfun": 1},
                    "constraints": None,
                    "jac": None,
                    "hess": None,
                    "hessp": None}
kwargs_pymoo = {"pop_size": 20,
                "sampling": "real_random",  # Notice that changing Hyper-Parameters may change pop size.
                "selection": "random",
                "crossover": "real_sbx",
                "mutation": "real_pm",
                "eliminate_duplicates": True,
                "n_offsprings": None}
```
 Merge the dictionaries into one.
 If you change the solver, also change the solver-kwargs-dict in the line below
```python
if framework == "scipy_differential_evolution":
    kwargs_optimization = kwargs_scipy_dif_evo
elif framework == "scipy_minimize":
    kwargs_optimization = kwargs_scipy_min
elif framework == "dlib_minimize":
    kwargs_optimization = kwargs_dlib_min
elif framework == "pymoo":
    kwargs_optimization = kwargs_pymoo
else:
    kwargs_optimization = {}
```
 Check if pymoo is being used for Multiprocessing
```python
if framework != "pymoo" and sim_api.n_cpu > 1:
    raise TypeError(f"Given framework {framework} does not support Multiprocessing."
                    f"Please use pymoo as your framework.")
```
 Select between single or multiple class calibration
```python
if isinstance(cal_classes, CalibrationClass):
    modelica_calibrator = Calibrator(
        cd=sim_api.cd,
        sim_api=sim_api,
        calibration_class=cal_classes,
        **kwargs_calibrator)
else:
    kwargs_calibrator.update(kwargs_multiple_classes)
    # Setup the class
    modelica_calibrator = MultipleClassCalibrator(
        cd=sim_api.cd,
        sim_api=sim_api,
        calibration_classes=cal_classes,
        start_time_method="fixstart",
        **kwargs_calibrator)
```
 ## Calibration
 Start the calibration process
```python
result = modelica_calibrator.calibrate(
    framework=framework,
    method=method,
    **kwargs_optimization
)
```
 ## Validation
 Start the validation process
```python
modelica_calibrator.validate(
    validation_class=validation_class,
    calibration_result=result
)
```
 Don't forget to close the simulation api:
```python
sim_api.close()
```
