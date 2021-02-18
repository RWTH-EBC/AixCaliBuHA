"""
Example file for the calibration package. The usage of modules and classes inside
the calibration package should be clear when looking at the examples.
If not, please raise an issue.
"""

import os
import numpy as np
from ebcpy.examples import dymola_api_example
from aixcalibuha.calibration import modelica
from aixcalibuha.examples import cal_classes_example
from aixcalibuha import CalibrationClass


def run_calibration(sim_api, cal_classes, stat_measure):
    """
    Run an example for a calibration. Make sure you have Dymola installed
    on your device and a working licence. All output data will be stored in
    the current working directory of python. Look at the logs and plots
    to better understand what is happening in the calibration. If you want, you
    can switch the methods to other supported methods or change the framework and
    try the global optimizer of dlib.

    :param ebcy.simulationapi.SimulationAPI sim_api:
        Simulation API to simulate the models
    :param list,CalibrationClass cal_classes:
        List with multiple CalibrationClass objects for calibration. Goals and
        TunerParameters have to be set. If only one class is provided (either
        a list with one entry or a CalibrationClass object) the single-class
        Calibrator is used.
    :param str stat_measure:
        Statistical measurement to evaluate the difference between simulated
        and real data.
    """
    # %% Settings:
    framework = "scipy_differential_evolution"
    method = "best1bin"
    # Specify values for keyword-arguments to customize the Calibration process for single-class
    kwargs_calibrator = {"timedelta": 0,
                         "save_files": False,
                         "verbose_logging": True,
                         "show_plot": True,
                         "create_tsd_plot": True,
                         "save_tsd_plot": True,
                         "fail_on_error": False,
                         "ret_val_on_error": np.NAN}
    # Specify kwargs for multiple-class-calibration
    kwargs_multiple_classes = {"merge_multiple_classes": True}

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
        kwargs_calibrator.update(kwargs_scipy_dif_evo)
    if framework == "scipy_minimize":
        kwargs_calibrator.update(kwargs_scipy_min)
    if framework == "dlib_minimize":
        kwargs_calibrator.update(kwargs_dlib_min)

    # Select between single or multiple class calibration
    if isinstance(cal_classes, CalibrationClass) or len(cal_classes)==1:
        modelica_calibrator = modelica.MultipleClassCalibrator(
            cd=sim_api.cd,
            sim_api=sim_api,
            statistical_measure=stat_measure,
            calibration_class=cal_classes,
            **kwargs_calibrator)
    else:
        kwargs_calibrator.update(kwargs_multiple_classes)
        # Setup the class
        modelica_calibrator = modelica.MultipleClassCalibrator(
            cd=sim_api.cd,
            sim_api=sim_api,
            statistical_measure=stat_measure,
            calibration_classes=cal_classes,
            start_time_method="timedelta",
            **kwargs_calibrator)

    # Start the calibration process
    modelica_calibrator.calibrate(framework=framework, method=method)


if __name__ == "__main__":
    # Parameters for calibration:
    STATISTICAL_MEASURE = "RMSE"

    CD = os.path.normpath(os.getcwd())

    DYM_API = dymola_api_example.setup_dymola_api(cd=CD)
    CAL_CLASSES = cal_classes_example.setup_calibration_classes()

    # %%Calibration:
    run_calibration(DYM_API,
                    CAL_CLASSES[2:4],
                    STATISTICAL_MEASURE)
