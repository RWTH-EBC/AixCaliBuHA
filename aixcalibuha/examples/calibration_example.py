"""
Example file for the calibration package. The usage of modules and classes inside
the calibration package should be clear when looking at the examples.
If not, please raise an issue.
"""

import os
from ebcpy.examples import dymola_api_example
from aixcalibuha.calibration import modelica
from aixcalibuha.examples import cal_classes_example


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
    :param list cal_classes:
        List with multiple CalibrationClass objects for calibration. Goals and
        TunerParameters have to be set.
    :param str stat_measure:
        Statistical measurement to evaluate the difference between simulated
        and real data.
    """
    # Specify values for keyword-arguments to customize the Calibration process
    kwargs_calibrator = {"save_files": False,
                         "save_tsd_plot": False,
                         "show_plot": True,
                         "verbose_logging": True}
    # Specify solver-specific keyword-arguments depending on the solver and method you will use
    kwargs_scipy_dif_evo = {"maxiter": 2,
                            "popsize": 2}
    kwargs_dlib_min = {"num_function_calls": 3}
    kwargs_scipy_min = {}

    # Merge the dictionaries into one.
    # If you change the solver, also change the solver-kwargs-dict in the line below
    kwargs_calibrator.update(kwargs_dlib_min)

    # Setup the class
    multiple_class_cal = modelica.MultipleClassCalibrator("dlib_minimize",
                                                          sim_api.cd,
                                                          sim_api,
                                                          stat_measure,
                                                          cal_classes,
                                                          start_time_method="timedelta",
                                                          reference_start_time=0,
                                                          **kwargs_calibrator)
    # Start the calibration process
    multiple_class_cal.calibrate(method="best1bin")


if __name__ == "__main__":
    # Parameters for calibration:
    # TODO: make single class calibration example
    STATISTICAL_MEASURE = "RMSE"

    CD = os.path.normpath(os.getcwd())

    DYM_API = dymola_api_example.setup_dymola_api()
    CAL_CLASSES = cal_classes_example.setup_calibration_classes()

    # %%Calibration:
    run_calibration(DYM_API,
                    CAL_CLASSES,
                    STATISTICAL_MEASURE)
