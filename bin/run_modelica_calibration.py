#!/usr/bin/env python
"""
Module to automatically run a calibration process
using modelica models.
You may use yml config files to alter the settings
in this file. The rest is done automatically.
Run the calibration after install of aixcalibuha using:
'modelica_calibration'
"""
import sys
import os
from ebcpy.utils import conversion
from ebcpy import data_types
from ebcpy.utils import configuration
from aixcalibuha.calibration import modelica
import aixcalibuha.utils.configuration as default_settings
from aixcalibuha.utils.configuration import get_calibration_classes_from_config

def _handle_argv(argv):
    """
    Supported argument are:

    Configuration of settings used in calibration:
    --config="Path_to_a_.yml_config_file"

    """
    # List of supported config options
    __supported_argv = ["--config"]
    # Path to the script is irrelevant
    resulting_setting = {}
    _rel_argv = argv[1:]
    for arg in _rel_argv:
        try:
            name, value = arg.split("=")
            if name not in __supported_argv:
                raise KeyError
            else:
                resulting_setting[name] = value
        except (ValueError, KeyError):
            raise ValueError(f"Given argument is not supported.\n{_handle_argv.__doc__}")

    return resulting_setting


def main():
    settings = _handle_argv(sys.argv)
    # Load config:
    config = configuration.read_config(settings["--config"])
    # Read all the relevant settings. If some are missing, rais an error:
    try:
        # First read necessary settings
        cd = config["Working Directory"]
        # Read calibration settings
        config_calibration = config["Calibration"]
        cal_classes_config = config_calibration["calibration_classes"]
        start_time_method = config_calibration.get("start_time_method", None)

        # Read optimization settings
        config_opt = config["Optimization"]
        framework = config_opt["framework"]
        method = config_opt["method"]

        # Read simulation settings
        config_sim = config["SimulationAPI"]
        if "cd" not in config_sim:
            config_sim["cd"] = cd
        _ = config_sim["type"]
        _ = config_sim["model_name"]
        _ = config_sim["packages"]

        # Read data for input values
        config_inp = config.get("Input Data", None)
        if config_inp is not None:
            sim_data_path = config_inp["sim_data_path"]
            sim_input_names = config_inp["sim_input_names"]
            meas_input_names = config_inp["meas_input_names"]
            meas_input_data = config_inp["meas_input_data"]
    except KeyError as e:
        value_error = str(e).replace("KeyError:", "")
        raise KeyError(f"Given config file does not contain {value_error}. "
                       f"You need to specify this value in order to run the calibration.")

    # Now set redundant parameters. If they are not in the settings it's ok.
    kwargs_calibrator = config_calibration.get("settings", default_settings.kwargs_calibrator)
    if len(cal_classes_config) > 1:
        config_cal_mul = config_calibration.get("settings", {})
        kwargs_multiple_classes = config_cal_mul.get("Multiple Classes", default_settings.kwargs_multiple_classes)
        kwargs_calibrator.update(kwargs_multiple_classes)

    # Specify solver-specific keyword-arguments depending on the solver and method you will use
    config_opt_kwargs = config_opt.get("settings", {})
    kwargs_scipy_dif_evo = config_opt_kwargs.get("scipy_differential_evolution",
                                                 configuration.kwargs_scipy_dif_evo)
    kwargs_dlib_min = config_opt_kwargs.get("dlib_minimize", configuration.kwargs_dlib_min)
    kwargs_scipy_min = config_opt_kwargs.get("scipy_minimize", configuration.kwargs_scipy_min)
    # Merge the dictionaries into one.
    # If you change the solver, also change the solver-kwargs-dict in the line below
    if framework == "scipy_differential_evolution":
        kwargs_calibrator.update(kwargs_scipy_dif_evo)
    if framework == "scipy_minimize":
        kwargs_calibrator.update(kwargs_scipy_min)
    if framework == "dlib_minimize":
        kwargs_calibrator.update(kwargs_dlib_min)

    # Start simulation api
    sim_api = configuration.get_simulation_api_from_config(config_sim)

    # Set the input data path:
    if config_inp is not None:
        map = {hdf_inp: sim_input_names[i] for i, hdf_inp in enumerate(meas_input_names)}
        # Load and rename columns so conversion works
        tsd = data_types.TimeSeriesData(**meas_input_data)
        _temp_path = os.path.join(cd, "temp.hdf")
        tsd.rename(columns=map).save(_temp_path, key="temp")
        # Create directories:
        dirpath = os.path.dirname(sim_data_path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        conversion.convert_hdf_to_modelica_txt(filepath=_temp_path,
                                               table_name="inputsMeasured",
                                               save_path_file=sim_data_path,
                                               columns=sim_input_names,
                                               key="temp",
                                               with_tag=False)
        # Delete generated file
        os.remove(_temp_path)

    # Load CalibrationClass Settings.
    cal_classes = get_calibration_classes_from_config(cal_classes_config)

    if len(cal_classes) == 1:
        modelica_calibrator = modelica.Calibrator(
            cd=cd,
            sim_api=sim_api,
            calibration_class=cal_classes[0],
            **kwargs_calibrator)
    else:
        # Setup the class
        modelica_calibrator = modelica.MultipleClassCalibrator(
            cd=cd,
            sim_api=sim_api,
            calibration_classes=cal_classes,
            start_time_method=start_time_method
            **kwargs_calibrator)

    # Start the calibration process
    modelica_calibrator.calibrate(framework=framework, method=method)


if __name__ == "__main__":
    sys.argv = ["", r"--config=D:\pme-fwu\00_testzone\test_wrapper\calibration_config.yml"]
    main()
