"""
Module to with configs and functions to read configs for objects in this repository.
"""

import numpy as np
from ebcpy import data_types
from ebcpy.utils.configuration import default_config, tsd_config
from aixcalibuha import Goals, CalibrationClass, TunerParas

# pylint: disable=line-too-long
kwargs_calibrator = {"timedelta": 0,
                     "save_files": False,
                     "verbose_logging": True,
                     "show_plot": True,
                     "create_tsd_plot": True,
                     "save_tsd_plot": True,
                     "fail_on_error": False,
                     "ret_val_on_error": np.NAN}

# Specify kwargs for multiple-class-calibration
kwargs_multiple_classes = {"merge_multiple_classes": True,
                           "fix_start_time": 0,
                           "timedelta": 0}

default_input_config = {"sim_input_names": None,
                        "sim_data_path": None,
                        "meas_input_names": None,
                        "meas_input_data": tsd_config,
                        }

default_cal_class_config = {"name": "TODO: Specify the name of the calibration class",
                            "start_time": "TODO: Specify the start time of the class - e.g 0",
                            "stop_time": "TODO: Specify the end time of the class",
                            "goals": {"meas_target_data": tsd_config,
                                      "variable_names": "TODO: Specify variable names",
                                      "weightings": "TODO: Insert null if you don´t need special weightings. "
                                                    "Else specify which goal get´s which weighting through a list"},
                            "tuner_paras": {
                                "names": "TODO: Specify the names of the tuner parameters list",
                                "initial_values": "TODO: Specify the inital values of the tuner parameters list",
                                "bounds": "TODO: Specify the boundaries of the tuner parameters as a list of tuples"}}

default_calibration_config = {
                    "statistical_measure": "TODO: Specify the statistical "
                                           "measure for calibration (RMSE, MAE, etc.)",
                    "calibration_classes": [default_cal_class_config],
                    "start_time_method": 'fixstart',
                    "settings": kwargs_calibrator,
                    "settings multiple classes": kwargs_multiple_classes
                    }

default_config.update({
    "Input Data": default_input_config,
    "Calibration": default_calibration_config,
    })


def get_goals_from_config(config: dict):
    """
    Read the data for a Goals object.

    :param dict config:
        Config holding the following cols for
        - meas_target_data
        - variable_names
        - Optional: weightings
    :return: Goals goals
        Loaded Goals object
    """
    config_mtd = config["meas_target_data"]
    mtd = data_types.TimeSeriesData(**config_mtd)
    return Goals(meas_target_data=mtd,
                 variable_names=config["variable_names"],
                 weightings=config.get("weightings", None))


def get_tuner_paras_from_config(config: dict):
    """
    Read the data for a TunerParas object.

    :param dict config:
        Config holding the following cols for
        - names
        - initial_values
        - bounds
    :return: TunerParas tuner_paras
        Loaded Goals object
    """
    return TunerParas(names=config["names"],
                      initial_values=config["initial_values"],
                      bounds=config["bounds"])


def get_calibration_classes_from_config(configs: list):
    """
    Read the data for a CalibrationClass object.

    :param list configs:
        List of dicts with configs holding the following cols for
        - names
        - start_time
        - stop_time
        - Optional: goals, tuner_paras, relevant_intervals
    :return: TunerParas tuner_paras
        Loaded Goals object
    """
    cal_classes = []
    for cal_class_config in configs:
        goals, tuner_paras = None, None
        if "goals" in cal_class_config:
            goals = get_goals_from_config(cal_class_config["goals"])
        if "tuner_paras" in cal_class_config:
            tuner_paras = get_tuner_paras_from_config(cal_class_config["tuner_paras"])
        cal_classes.append(
            CalibrationClass(name=cal_class_config["name"],
                             start_time=cal_class_config["start_time"],
                             stop_time=cal_class_config["stop_time"],
                             goals=goals,
                             tuner_paras=tuner_paras,
                             relevant_intervals=cal_class_config.get("relevant_intervals", None)))
    return cal_classes
