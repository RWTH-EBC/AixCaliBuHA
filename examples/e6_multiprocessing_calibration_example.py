import numpy as np
from aixcalibuha import CalibrationClass, Calibrator, MultipleClassCalibrator


def run_calibration(sim_api, cal_classes, validation_class, n_cpu):
    """
    Make sure you are using pandas==1.3.5 and tables==3.6.1

    Run an example for a calibration using multiprocessing. Multiprocessing only runs with pymoo as the used framework.
    Make sure you have Dymola installedmon your device and a working licence. All output data will be stored in
    the current working directory of python. Look at the logs and plots
    to better understand what is happening in the calibration. If you want, you
    can switch the methods to other supported methods or change the framework and
    try the global optimizer of dlib.

    :param ebcy.simulationapi.SimulationAPI sim_api:
        Simulation API to simulate the models
    :param list[CalibrationClass] cal_classes:
        List with multiple CalibrationClass objects for calibration. Goals and
        TunerParameters have to be set. If only one class is provided (either
        a list with one entry or a CalibrationClass object) the single-class
        Calibrator is used.
    :param CalibrationClass validation_class:
        Class used to validate the findings.
    :param int n_cpu:
        Number of logical Processors to run calibration on.
    """
    # %% Settings:
    # if the selected framework is "pymoo", method is the used algorithm and will here be automatically set to "GA"
    framework = "pymoo"
    method = "GA"
    # Specify values for keyword-arguments to customize the Calibration process for single-class
    kwargs_calibrator = {"timedelta": 0,
                         "save_files": False,
                         "verbose_logging": True,
                         "show_plot": True,
                         "create_tsd_plot": True,
                         "save_tsd_plot": True,
                         "plot_file_type": "png",
                         "fail_on_error": False,
                         "ret_val_on_error": np.NAN,
                         "max_itercount": 10000
                         }
    # Specify kwargs for multiple-class-calibration
    kwargs_multiple_classes = {"merge_multiple_classes": True}

    # Specify solver-specific keyword-arguments depending on the solver and method you will use
    kwargs_scipy_dif_evo = {"maxiter": 100,  # maximum number of generations over which the entire population is evolved
                            "popsize": 15,
                            "mutation": (0.7, 1),
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
    kwargs_pymoo = {"n_cpu": n_cpu,
                    "pop_size": 100,
                    "sampling": "real_random",  # Notice that changing Hyperparameters may change pop size.
                    "selection": "random",
                    "crossover": "real_sbx",
                    "mutation": "real_pm",
                    "eliminate_duplicates": True,
                    "n_offsprings": None}

    # Merge the dictionaries into one.
    # If you change the solver, also change the solver-kwargs-dict in the line below
    if framework == "scipy_differential_evolution":
        kwargs_optimization = kwargs_scipy_dif_evo
    elif framework == "scipy_minimize":
        kwargs_optimization = kwargs_scipy_min
    elif framework == "dlib_minimize":
        kwargs_optimization = kwargs_dlib_min
    elif framework == "pymoo":
        kwargs_optimization = kwargs_pymoo
        method = "GA"
    else:
        kwargs_optimization = {}
    # Check if pymoo is being used for Multiprocessing
    if not framework == "pymoo" and n_cpu > 1:
        raise TypeError(f"Given framework {framework} does not support Multiprocessing."
                        f" Please use pymoo as your framework.")
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

    # Start the calibration process
    result = modelica_calibrator.calibrate(
        framework=framework,
        method=method,
        **kwargs_optimization
    )
    modelica_calibrator.validate(
        validation_class=validation_class,
        tuner_parameter_values=list(result.values())
    )
    # Don't forget to close the simulation api
    sim_api.close()

if __name__ == "__main__":
    from examples import setup_fmu, setup_calibration_classes
    # n_cpu:
    n_cpu = 2
    # Parameters for sen-analysis:
    EXAMPLE = "A"  # Or choose B
    SIM_API = setup_fmu(example=EXAMPLE, n_cpu=n_cpu)
    CALIBRATION_CLASSES, VALIDATION_CLASS = setup_calibration_classes(
        example=EXAMPLE,
        multiple_classes=False
    )

    # Sensitivity analysis:
    run_calibration(
        sim_api=SIM_API,
        cal_classes=CALIBRATION_CLASSES,
        validation_class=VALIDATION_CLASS,
        n_cpu=n_cpu
    )