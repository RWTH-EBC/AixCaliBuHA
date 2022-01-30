# # Example 3 sensitivity analysis

# Goals of this part of the examples:
# 1. Learn the settings for a calibration
# 2. Learn how to use both Single- and MultiClassCalibration
# 3. Learn how to validate your calibration

import numpy as np
from aixcalibuha import CalibrationClass, Calibrator, MultipleClassCalibrator


def run_calibration(example="B", sim_api=None, cal_classes=None):
    """
    Run an example for a calibration. Make sure you have Dymola installed
    on your device and a working licence. All output data will be stored in
    the current working directory of python. Look at the logs and plots
    to better understand what is happening in the calibration. If you want, you
    can switch the methods to other supported methods or change the framework and
    try the global optimizer of dlib.

    :param str example:
        Which example to run, "A" or "B"

    Optional, for the fully automated process
    :param ebcy.simulationapi.SimulationAPI sim_api:
        Simulation API to simulate the models
    :param list[CalibrationClass] cal_classes:
        List with multiple CalibrationClass objects for calibration. Goals and
        TunerParameters have to be set. If only one class is provided (either
        a list with one entry or a CalibrationClass object) the single-class
        Calibrator is used.
    """

    # ## Setup
    # Start by loading the simulation api and the calibration classes
    from examples import setup_fmu, setup_calibration_classes
    if sim_api is None:
        sim_api = setup_fmu(example=example)
    default_cal_classes, validation_class = setup_calibration_classes(
        example=example,
        multiple_classes=False
    )
    if cal_classes is None:
        cal_classes = default_cal_classes

    # ## Calibration and optimization settings
    # We refer to the docstrings on more information on each setting.
    # Specify values for keyword-arguments to customize
    # the Calibration process for a single-class calibration
    kwargs_calibrator = {"timedelta": 0,
                         "save_files": False,
                         "verbose_logging": True,
                         "show_plot": True,
                         "create_tsd_plot": True,
                         "save_tsd_plot": True,
                         "plot_file_type": "png",
                         "fail_on_error": False,
                         "ret_val_on_error": np.NAN,
                         # For this example, let's keep the runtime low
                         "max_itercount": 100
                         }
    # Specify values for keyword-arguments to customize
    # the Calibration process for a multiple-class calibration
    kwargs_multiple_classes = {"merge_multiple_classes": True}

    # Specify the framework and the method for the underlying optimization:
    framework = "scipy_differential_evolution"
    method = "best1bin"

    # Specify solver-specific keyword-arguments depending on the solver and method you will use
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

    # Merge the dictionaries into one.
    # If you change the solver, also change the solver-kwargs-dict in the line below
    if framework == "scipy_differential_evolution":
        kwargs_optimization = kwargs_scipy_dif_evo
    elif framework == "scipy_minimize":
        kwargs_optimization = kwargs_scipy_min
    elif framework == "dlib_minimize":
        kwargs_optimization = kwargs_dlib_min
    else:
        kwargs_optimization = {}
    # Select between single or multiple class calibration
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

    # ## Calibration
    # Start the calibration process
    result = modelica_calibrator.calibrate(
        framework=framework,
        method=method,
        **kwargs_optimization
    )

    # ## Validation
    # Start the validation process
    modelica_calibrator.validate(
        validation_class=validation_class,
        tuner_parameter_values=list(result.values())
    )
    # Don't forget to close the simulation api:
    sim_api.close()


if __name__ == "__main__":
    # Parameters for sen-analysis:
    EXAMPLE = "B"  # Or choose A
    # Sensitivity analysis:
    run_calibration(example=EXAMPLE)
