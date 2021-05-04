#!/usr/bin/env python
"""
Module to automatically setup a calibration process
using modelica models.
"""
import os
import sys
from ebcpy.data_types import TimeSeriesData
from ebcpy.simulationapi.dymola_api import DymolaAPI
from ebcpy.modelica import get_names_and_values_of_lines, get_expressions
from ebcpy.utils import configuration
from aixcalibuha.utils import configuration as defaultsettings


def _handle_tsd_input():
    key = None
    sheet_name = None
    while True:
        print("NOTE: csv files are loaded with ',' separator. \n"
              "      Change your file manually if this is not the case for your file.")
        fpath = input("Place your file below and press ENTER:")
        if not os.path.exists(fpath):
            print("File not found on your machine. Try again.")
            continue
        try:
            TimeSeriesData(fpath, key=key, sheet_name=sheet_name)
        except KeyError as e:
            print(e)
            if fpath.endswith(".hdf"):
                key = input("Pass the key of the hdf and press ENTER:")
            elif fpath.endswith(".xlsx"):
                sheet_name = input("Pass the name of the sheet and press ENTER:")
            continue
        finally:
            break

    return {"data": fpath,
            "key": key,
            "sheet_name": sheet_name}


def _handle_col_matching(tsd, cols):
    columns = tsd.columns.get_level_values(0)
    print(f"Columns of the file:\n {columns.values}")
    cols_matches = []
    for col in cols:
        while True:
            matcher = input(f"{col} = ")
            if matcher not in columns:
                print("Given name not in columns listed above. Try again:")
            else:
                break
        cols_matches.append(matcher)
    return cols_matches


def _handle_argv(argv):
    """
    Supported argument are:

    Configuration of settings for the setup:
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


def _automatically_generate_mocate(model_name, savepath, packages=None, **kwargs):
    """
    Function to generate a MOdelica CAlibration TEmplates (MoCaTe)
    package for the given model inside the simulation api.

    :param str model_name:
        Name of the model you want to calibrate.
        If the model is a standalone model without a "package.mo" file,
        you have to pass the path the model. Else the path is extracted
        from the packages-list (see below)
    :param os.path.normpath,str savepath:
        Path to store the generated package
    :param list packages:
        List with filepath to the "package.mo" file of each package required
        to simulate the given model_name.
    :keyword str mocate_version
        Version number of used Modelica Calibration Templates Library. Default is 0.0.4
    :keyword str msl_version
        Version number of used Modelica Calibration Templates Library. Default is 3.2.3
    :return:
    """
    mocate_version = kwargs.pop("mocate_version", "0.0.4")
    msl_version = kwargs.pop("msl_version", "3.2.3")

    # Create save-path folder if necessary
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    # First extract the model-name to calibrate:
    if packages is None:
        if not os.path.exists(model_name):
            raise FileExistsError(f"Given model_name path {model_name} could not be "
                                  f"found on your machine.")
        if not os.path.isfile(model_name):
            raise FileNotFoundError(f"Given model_name {model_name} is not a valid file.")
        _model_name = os.path.split(model_name)[1].split(".")[0]
        _model_name_os = model_name
        _model_name_modelica_long = _model_name
    else:
        _model_name_modelica_long = model_name
        _model_name = model_name.split(".")[-1]
        _model_name_os = None
        _temp_package_name = model_name.split(".")[0]
        # Extract filepath of model and check correct input of packages:
        for package in packages:
            if not package.endswith("package.mo"):
                raise ValueError(f"packages contains a filepath not "
                                 f"pointing at a package.mo file: {package}")
            if _temp_package_name in package:
                _model_name_os = os.path.normpath(
                    os.path.join(os.path.dirname(package),
                                 "\\".join(model_name.split(".")[1:]) + ".mo"))
        if _model_name_os is None:
            raise ValueError("Given packages do not include the given model_name.")

    # Extract parameters from model and create record
    tuner_paras = get_expressions(filepath_model=_model_name_os, modelica_type="parameters")
    _tuner_paras_from_model = "\n  ".join(tuner_paras)
    # Extract only the variable names
    tuner_para_names = [tuner_para.split("=")[0].split()[-1]
                        for tuner_para in tuner_paras]
    _tuner_paras_written_to_model_container = ",\n        ".join(
        [f"final {tuner_para}="f"tunerParameters.{tuner_para}"
         for tuner_para in tuner_para_names])

    # Create package-folder and files:

    # Parent directory:
    _package_name = f"{_model_name}AixCalibration"
    _package_path = os.path.join(savepath, _package_name)
    os.mkdir(_package_path)
    # Write files:
    with open(os.path.join(_package_path, "package.mo"), "a+") as file:
        file.write(f'within ;\npackage {_package_name}\n\nannotation (uses(Modelica('
                   f'version="{msl_version}"), CalibrationTemplates(versi'
                   f'on="{mocate_version}")));\nend {_package_name};')
    with open(os.path.join(_package_path, "package.order"), "a+") as file:
        file.write(f"{_model_name}Simulator\n{_model_name}Adapted\nDatabase\nInterfaces")
    with open(os.path.join(_package_path, f"{_model_name}Simulator.mo"), "a+") as file:
        file.write(f'within {_package_name};\nmodel {_model_name}Simulator\n  '
                   f'extends CalibrationTemplates.SimulatorTemplates.Internal.'
                   f'PartialAixCaliBuHaSimulator(\n    redeclare '
                   f'{_model_name}Adapted modelContainer(\n        '
                   f'redeclare Interfaces.CalBusTargetsSimed '
                   f'calBusTargetSimed,\n        redeclare Interfaces.'
                   f'CalBusInputs calBusInput,\n        '
                   f'{_tuner_paras_written_to_model_container}),\n    redeclare '
                   f'Database.TunerParameters tunerParameters,\n    '
                   f'redeclare Interfaces.CalBusInputs calBusInput,\n'
                   f'    redeclare Interfaces.CalBusTargetsSimed '
                   f'calBusTargetSimed);\n\nend {_model_name}Simulator;')
    with open(os.path.join(_package_path, f"{_model_name}Adapted.mo"), "a+") as file:
        file.write(f'within {_package_name};\nmodel {_model_name}Adapted\n  '
                   f'extends CalibrationTemplates.Interfaces.Containers.ModelContainer(\n  '
                   f'redeclare Interfaces.CalBusInputs calBusInput,\n  '
                   f'redeclare Interfaces.CalBusTargetsSimed calBusTargetSimed);\n  '
                   f'extends {_model_name_modelica_long};\n'
                   f'\nend {_model_name}Adapted;')

    # Database
    os.mkdir(os.path.join(_package_path, "Database"))
    with open(os.path.join(_package_path, "Database", "package.mo"), "a+") as file:
        file.write(f"within {_package_name};\npackage Database\nend Database;\n")
    with open(os.path.join(_package_path, "Database", "package.order"), "a+") as file:
        file.write("TunerParameters\n")
    with open(os.path.join(_package_path, "Database", "TunerParameters.mo"), "a+") as file:
        file.write(f'within {_package_name}.Database;\nrecord TunerParameters '
                   f'"Created by AixCaliBuHa"\n  extends CalibrationTemplates.'
                   f'Database.TunerParameterBaseDataDefinition;\n'
                   # Created based on the parameters in the given model:
                   f'  {_tuner_paras_from_model}\n'
                   f'end TunerParameters;')

    # Interfaces
    os.mkdir(os.path.join(_package_path, "Interfaces"))
    with open(os.path.join(_package_path, "Interfaces", "package.mo"), "a+") as file:
        file.write(f"within {_package_name};\npackage Interfaces\n  "
                   f"extends Modelica.Icons.InterfacesPackage;\nend Interfaces;\n")
    with open(os.path.join(_package_path, "Interfaces", "package.order"), "a+") as file:
        file.write("CalBusInputs\nCalBusTargetsSimed\n")
    with open(os.path.join(_package_path, "Interfaces", "CalBusInputs.mo"), "a+") as file:
        file.write(f'within {_package_name}.Interfaces;\nexpandable connector '
                   f'CalBusInputs\n  "Add the inputs to your model here"\n  '
                   f'extends CalibrationTemplates.Interfaces.CalBusInputs;\n\n  '
                   f'//Specify a new variable for each input of the system you want\n  '
                   f'//to provide to simulate your system.\n  '
                   f'//Example:\n  '
                   f'//Real rotational_speed "Rotational speed of a device in the system";\n'
                   f'\n\nend CalBusInputs;')
    with open(os.path.join(_package_path, "Interfaces",
                           "CalBusTargetsSimed.mo"), "a+") as file:
        file.write(f'within {_package_name}.Interfaces;\nexpandable connector '
                   f'CalBusTargetsSimed\n  "Add the target values simulated of '
                   f'your calibration here"\n  extends CalibrationTemplates.'
                   f'Interfaces.CalBusTargetsSimed;\n\n  '
                   f'//Specify a new variable for each target (output) of the system you want\n  '
                   f'//to compare with measured data in AixCaliBuHa.\n  '
                   f'//Example:\n  '
                   f'//Modelica.SIunits.Temperature T_out "Output temperature of a system";\n'
                   f'\n\nend CalBusTargetsSimed;')

    # Return the package.mo path to better load the model
    return os.path.join(_package_path, "package.mo"), tuner_paras


def main():
    """
    How does this look?
    :return:
    """
    print("----------------------Setup of Dymola--------------------")
    if sys.argv[1:]:
        settings = _handle_argv(sys.argv)
        config = configuration.read_config(settings["--config"])
        _cd = config["cd"]
        model_name = config["model_name"]
        packages = config["packages"]
    else:
        model_name = input("\nStep 1/3:\n"
                           "What model do you want to calibrate?\n"
                           "Please pass the complete Modelica path,\n"
                           " e.g. 'MyLibrary.Componentes.MyModel'. \n"
                           "Afterwards, hit ENTER!\n")

        print("\n\nStep 2/3:\nIn order to automatically anaylze the model, \n"
              "you also have to pass the path to the package.mo \n"
              "of all libraries the model depends on (except standard libraries),\n"
              " e.g. 'path_to_my_library/package.mo'.\n"
              "If you are finished adding packages, type [stop] and press ENTER!\n")

        packages = []
        while True:
            pkg = input("Path to package.mo: ")
            if pkg.lower() == "stop":
                break
            elif not os.path.isfile(pkg):
                print("File does not exist on your machine! Try again.")
            else:
                packages.append(pkg)

        _cd = input("\n\nStep 3/3:\nWhere do you want the output of the setup to be stored?\n"
                    "Either pass a full path or a directory relative to your current directory.\n"
                    "e.g. 'C://path_to_be_stored' or 'relative_path'. \n"
                    "If you pass nothing, the current directoy will be used.\n"
                    "Afterwards, hit ENTER!\n")

    # Convert path input
    if os.path.isabs(_cd):
        cd = _cd
    else:
        cd = os.path.join(os.getcwd(), _cd)

    # Save this settings:
    setup_config = {"cd": cd,
                    "model_name": model_name,
                    "packages": packages.copy()}
    try:

        # Make dirs if necessary
        if not os.path.exists(cd):
            os.mkdir(cd)

        _real_model_name = model_name.split(".")[-1]
        if f"{_real_model_name}AixCalibration" in os.listdir(cd):
            raise FileExistsError(f"{_real_model_name}AixCalibration already exisits in"
                                  f" given path. Please delete the package or change the directory")

        mocate_adapted_model, tuner_paras_codelines = \
            _automatically_generate_mocate(model_name=model_name,
                                           savepath=cd,
                                           packages=packages)

        packages.append(mocate_adapted_model)
        # TODO: Check if this still works for new installs
        _mocate_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    "modelica_calibration_templates",
                                    "CalibrationTemplates",
                                    "package.mo")

        packages.append(_mocate_path)

        dym_api = DymolaAPI(cd=cd, model_name=model_name, packages=packages, show_window=True)

        print("\n----------------------Model Adjustments--------------------")
        _model_name_internal = f"{_real_model_name}AixCalibration.{_real_model_name}Adapted"

        _cur_model = f"{_real_model_name}AixCalibration.Interfaces.CalBusInputs"
        dym_api.dymola.openModelFile(_cur_model)
        print("\nStep 0/3:\nWe already did this one for you - just an info:\n"
              "Based on your model, all top-level parameters were extracted "
              "and stored in the record under `Database.TunerParameters`.\n"
              "These values will be saved later to the config.\n You may remove "
              "parameters you don't want to calibrate from the config list once we are finished.\n\n")

        input("\nStep 1/3:\nLet's define the inputs to the system.\n"
              "We already opened the required connector Interfaces.CalBusInputs for you.\n"
              "Go into the code editor in Dymola (the 'Modelica Text' section) and add them.\n"
              "You will find an example on how to define a variable in the code section.\n"
              "After you are done, save the model and press ENTER!")
        while not dym_api.dymola.checkModel(_cur_model):
            input("Check of model failed. After you correct the error, press ENTER.")

        _cur_model = f"{_real_model_name}AixCalibration.Interfaces.CalBusTargetsSimed"
        dym_api.dymola.openModelFile(_cur_model)
        input("\n\nStep 2/3:\nNow we will define the targets of our calibration.\n"
              "We already opened the required connector Interfaces.CalBusTargetsSimed for you.\n"
              "Now go into the code editor in Dymola (the 'Modelica Text' section) and add them.\n"
              "You will find an example on how to define a variable in the code section.\n"
              "After you are done, save the model and press ENTER!")
        while not dym_api.dymola.checkModel(_cur_model):
            input("Check of model failed. After you correct the error, press ENTER.")

        dym_api.dymola.openModelFile(_model_name_internal)
        input("\n\nStep 3/3:\n"
              "At last we opened the model to be calibrated for you.\n"
              "Now, connect the inputs and targets values of your system \n"
              "to the respective bus connectors you recently filled with new variables. \n\n"
              "NOTE: If you have to make adjustments on the model:\n"
              "Open the original model, change it and save it.\n"
              "As we extend from the original model you are only "
              "able to connect new things to interfaces already existing.\n"
              "After you are done, save the model and press ENTER!")
        while not dym_api.dymola.checkModel(_cur_model):
            input("Check of model failed. After you correct the error, press ENTER.")

        # Read Input Names
        path_to_input_names = os.path.join(cd, f"{_real_model_name}AixCalibration", "Interfaces",
                                           "CalBusInputs.mo")
        # Later used to store the .txt. file:
        path_to_input_data = os.path.join(cd, f"{_real_model_name}AixCalibration", "Data",
                                          "Inputs", "inputs_measured.txt")
        input_names_codelines = get_expressions(filepath_model=path_to_input_names,
                                                modelica_type="variables")

        # Make connections for input-variables
        input_names = [res["name"] for res in get_names_and_values_of_lines(input_names_codelines)]

        if not input_names:
            raise ValueError("You need to specify inputs to the system. Else this setup won't work")

        input_names_str = '{"%s"}' % '", "'.join(input_names)
        with open(os.path.join(cd, f"{_real_model_name}AixCalibration",
                               f"{_real_model_name}Simulator.mo"), "r+") as file:
            lines = file.readlines()
            file.seek(0)
            file.truncate()
            lines.insert(3, f"    final inputNames={input_names_str},\n")
            lines.insert(3, f"    final fNameInputsMeas=fNameInputsMeas_internal,\n")
            lines.insert(-2, f'  parameter String fNameInputsMeas_internal='
                             f'"{DymolaAPI._make_modelica_normpath(path_to_input_data)}";\n')
            lines.insert(-2, "equation\n")
            for i, input_name in enumerate(input_names):
                lines.insert(-2, f"  connect(tableInputsMeas.y[{i + 1}], calBusInput.{input_name});\n")
            file.writelines(lines)

        # Reload the package
        dym_api.dymola.openModel(mocate_adapted_model, changeDirectory=False)
        # Open the model and check it:
        _cur_model = f"{_real_model_name}AixCalibration.{_real_model_name}Simulator"
        dym_api.dymola.openModelFile(_cur_model)
        while not dym_api.dymola.checkModel(_cur_model):
            input("\n\nFor some reason, the automatically generated model did not check.\n"
                  "Please try to find the error and correct it.\n"
                  "Afterwards, press Enter.\n "
                  "If you don't want to correct the error, write 'stop' and press Enter")

        # The simulator model is the one we calibrate at the end
        model_name_for_calibration = _cur_model

        # Generate new python-script for automatic calibration:
        # Read Target Names
        path_to_target_names = os.path.join(cd, f"{_real_model_name}AixCalibration", "Interfaces",
                                            "CalBusTargetsSimed.mo")
        target_names_codelines = get_expressions(filepath_model=path_to_target_names,
                                                 modelica_type="variables")
        tuner_paras_var = get_names_and_values_of_lines(tuner_paras_codelines)
        tuner_paras_names = ["tunerParameters." + var["name"] for var in tuner_paras_var]
        tuner_paras_values = [var["value"] for var in tuner_paras_var]
        target_names = ["calBusTargetSimed." + var["name"]
                        for var in get_names_and_values_of_lines(target_names_codelines)]

        if not target_names:
            raise ValueError("You need to specify targets of the system. Else a calibration won't work")

        print("\n----------------------Provide Target Measurements--------------------")
        print("\n\nStep 1/2:\n"
              "To compare your model with measured data, \n"
              "you have to provide the files where your target measurements are stored.\n"
              "We support time series data in formats .hdf, .mat, .csv and .xlsx.\n")
        kwargs_tar = _handle_tsd_input()
        tsd = TimeSeriesData(**kwargs_tar)
        print("\n\nStep 2/2:\n"
              "Now specify which columns of the data you want to match to the goals:")
        keys_matches = _handle_col_matching(tsd, cols=target_names)

        # Convert config.
        variable_name_config = {}
        for target_name_sim, target_name_meas in zip(target_names, keys_matches):
            variable_name_config[target_name_sim] = [target_name_meas, target_name_sim]

        # First generate default config:
        savepath = os.path.join(cd, "calibration_config.yml")
        configuration.write_config(savepath, defaultsettings.default_config)

        cal_class_config = defaultsettings.default_cal_class_config
        cal_class_config["tuner_paras"] = {
            "names": tuner_paras_names,
            "initial_values": tuner_paras_values,
            # Assume a range of min and max values to make the start with the calibration more easier
            "bounds": [[val * 0.1, val * 2] for val in tuner_paras_values]}
        cal_class_config["goals"] = {
            "variable_names": variable_name_config,
            "meas_target_data": kwargs_tar,
            "weightings": [1 / len(target_names) for _ in target_names]}

        print("\n----------------------Provide Input Measurements--------------------")
        print("\n\nStep 1/2:\n"
              "To simulate your model with the correct boundary conditions, \n"
              "you have to provide the files where your input measurements are stored.\n"
              "We support time series data in formats .hdf, .mat, .csv and .xlsx.\n")
        kwargs_inp = _handle_tsd_input()
        tsd = TimeSeriesData(**kwargs_inp)
        print("\n\nStep 2/2:\n"
              "Now specify which columns of the data you want to match to the inputs:")
        col_matches_inp = _handle_col_matching(tsd, cols=input_names)

        # Then update old config by the recursive write_config function in configuration.py
        config = {
            "SimulationAPI": {"packages": packages,
                              "model_name": model_name_for_calibration,
                              "type": "DymolaAPI",
                              "dymola_path": dym_api.dymola_path,
                              "dymola_interface_path": dym_api.dymola_interface_path},
            "Calibration": {
                "calibration_classes": [cal_class_config]
            },
            "Input Data": {"sim_input_names": input_names,
                           "sim_data_path": path_to_input_data,
                           "meas_input_names": col_matches_inp,
                           "meas_input_data": kwargs_inp,
                           },
            "Working Directory": cd
        }

        configuration.write_config(savepath, config)
        print("\n----------------------Config Adjustments--------------------")

        print("\nYou may now check the output and make adjustments in the config:\n"
              f"{savepath}.\n"
              f"\nYou will find 'TODOS' where you have to make further choices.\n"
              f"For a quick test, the following values should work.\n"
              f"Please read the doc of the python classes to better understand all the settings.\n"
              f"    calibration_classes:\n"
              f"        name: test_class\n"
              f"        start_time: 0\n"
              f"        stop_time: 10\n"
              f"    statistical_measure: NRMSE\n"
              f"    framework: scipy_differential_evolution\n"
              f"    method: best1bin\n"
              "\n----------------------Run Calibration--------------------\n"
              f"After you made the adjustments, you can run your calibration via:\n"
              f"    'modelica_calibration --config={savepath}'")
    except Exception as e:
        _temp_savepath = os.path.join(cd, "guided_setup_config.yml")
        configuration.write_config(_temp_savepath, setup_config)
        print("An error occured in the process.\n"
              f"{e}"
              "\nRestart the setup with the same values you entered via:\n"
              f"    'guided_setup --config={_temp_savepath}'")


if __name__ == "__main__":
    sys.argv = [""]
    main()
