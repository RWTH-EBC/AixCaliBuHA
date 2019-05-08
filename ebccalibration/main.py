"""Main file for coordination of all steps of a calibration.
E.g. Preprocessing, classifier, calibration etc."""
from docutils.nodes import contact

from ebccalibration.calibration import DymolaAPI
from ebccalibration.calibration import Calibrator
import os
# from ebcpython.modelica.tools import manipulate_dsin  # TODO provide manipulate_dsin in another project.


def continouus_calibration(continouus_data, type_of_continouus_calibration, dymola_api, work_dir, qual_meas, method, cal_kwargs, referenceTime = 0):
    """
    :param continouus_data:
    List with dictionaries used for continoous calibration
    :param type_of_continouus_calibration: str
    Three modes are possible: dsfinal, timedelta, fix_start. See \img\type_of_continouus_calibration.png for an overview.
    For the last two options, the parameter referenceTime either gives the timedelta before each simulation OR the fix_start time.
    :param dymola_api:
    Dymola API class
    :param work_dir: os.path.normpath
    :param qual_meas: str
    See Calibrator.Calibrator
    :param method: str
    See Calibrator.Calibrator
    :param cal_kwargs:
    See Calibrator.Calibrator
    :param referenceTime: float or int
    Used as a offset value for the simulation time OR as a fixed start_time.
    :return:
    """
    assert type_of_continouus_calibration in ["dsfinal", "timedelta", "fix_start"], "The given type of continouus calibration %s does not match a given choice."%type_of_continouus_calibration
    # Join all initial names:
    total_initial_names = Calibrator.join_tuner_paras(continouus_data)
    # Calibrate
    # manipulate_dsin.eliminate_parameters(r"D:\dsfinal.txt", r"D:\test.txt", [],eliminateAuxiliarParmateres=True)
    cal_history = []
    curr_num = 0
    for c in continouus_data:
        # Alter the simulation time. This depends on the mode one is using.
        if type_of_continouus_calibration == "timedelta":
            start_time = float(c["start_time"]-referenceTime)
        elif type_of_continouus_calibration == "fix_start":
            start_time = float(referenceTime)
        elif type_of_continouus_calibration == "dsfinal":
            start_time = float(c["start_time"])
        dymola_api.set_sim_setup({"startTime": start_time,
                             "stopTime": float(c["stop_time"])})
        # Alter the working directory for the simulations
        cwdir_of_class = os.path.join(work_dir, "%s_%s"%(curr_num, c["class"]))
        dymola_api.set_cwdir(cwdir_of_class)
        # Alter tunerParas based on old results
        if len(cal_history) > 0:
            tuner_para = Calibrator.alter_tuner_paras(c["tuner_para"], cal_history)
            # Alter the dsfinal for the new phase
            if type_of_continouus_calibration == "dsfinal":
                new_dsfinal = os.path.join(dymola_api.cwdir, "dsfinal.txt")
                manipulate_dsin.eliminate_parameters(
                    os.path.join(cal_history[-1]["cal"].savepath_min_result, "dsfinal.txt"), new_dsfinal,
                    total_initial_names)
                dymola_api.import_initial(new_dsfinal)
        else:
            tuner_para = c["tuner_para"]
        # Create class with new simulationapi
        print("Starting with class {} in the time period: start = {} to end = {}".format(c["class"], c["start_time"], c["stop_time"]))
        cal = Calibrator.Calibrator(goals=c["goals"],
                                    tuner_para=tuner_para,
                                    qual_meas=qual_meas,
                                    method=method,
                                    dymola_api=dymola_api,
                                    time_info_tupel= (c["start_time"], c["stop_time"]),
                                    **cal_kwargs)
        res = cal.calibrate(cal.objective)
        if hasattr(cal, "traj_names"):
            total_initial_names = list(set(total_initial_names + cal.traj_names))
        cal_history.append({"cal": cal,
                           "res": res,
                           "continouus_data": c})
        curr_num += 1

    dymola_api.dymola.close()
    print("Final parameter values after calibration:")
    print(Calibrator._get_continouus_averages(cal_history))


def example(continouus = False):
    """Example function for a calibration process"""
    input_path = input("Please enter the directory to excecute and save the results of this example:")
    working_dir = os.path.normpath(input_path)
    # Declaring goals and tuners:
    # Measurement values must be passed through the Modelica simulation.
    goals = [{"meas": "trap_meas",
              "meas_full_modelica_name" : "trapezoid.y",
              "sim": "sim",
              "sim_full_modelica_name" : "sine.y",
              "weighting": 0.8},
             {"meas": "pulse_meas",
              "meas_full_modelica_name" : "pulse.y",
              "sim": "sim",
              "sim_full_modelica_name" : "sine.y",
              "weighting": 0.2}]
    tuner_para = {"amplitude": {"start": 2, "uppBou": 3, "lowBou": 0.3},
                 "freqHz": {"start": 0.5, "uppBou": 0.99, "lowBou": 0.001}}
    # Save the dictionaries to xml--> Just for showing how to workflow will be
    goal_xml = os.path.join(working_dir, "goalTest.xml")
    tuner_xml = os.path.join(working_dir, "tunerTest.xml")
    Calibrator.save_xml(goal_xml, goals)
    Calibrator.save_xml(tuner_xml, tuner_para)
    # Reload them--> Just for showing how to workflow will be
    goals = Calibrator.load_goals_xml(goal_xml)
    tuner_para = Calibrator.load_tuner_xml(tuner_xml)
    if continouus:
        continouus_data = [{"start_time": 0.0,
                           "stop_time": 10.0,
                           "class": "Anschalten",
                           "goals": goals,
                           "tuner_para": tuner_para},
                           {"start_time": 10.0,
                           "stop_time": 20.0,
                           "class": "Ausschalten",
                           "goals": goals,
                           "tuner_para": tuner_para}]
    # Setup simulationapi
    ex_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "calibration", "examples", "ExampleCalibration.mo")
    packages = [os.path.normpath(ex_path)]
    dymola_api = DymolaAPI.DymolaAPI(working_dir, packages,
                                       "ExampleCalibration")
    method_options = {"maxiter": 100000,       # Maximal iterations. Abort after maxiter even if no minimum has been achieved.
               "disp": False,           # Show additional infos in console
               "ftol": 2.220446049250313e-09,
               "eps": 0.1
               }
    cal_kwargs = {"method_options": method_options,
              #"tol": 0.95,              # Overall objective function tolerance, e.g. minimize until RMSE < 0.95
              "plotCallback": True,
              "save_files": False,
              "continouus_calibration": continouus}
    quality_measure = "NRMSE"
    optimizer_method = "L-BFGS-B"

    if continouus:
        continouus_calibration(continouus_data=continouus_data,
                               type_of_continouus_calibration = "timedelta",
                               dymola_api=dymola_api,
                               work_dir=working_dir,
                               qual_meas=quality_measure,
                               method=optimizer_method,
                               cal_kwargs=cal_kwargs,
                               referenceTime=0)
    else:
        dymola_api.set_sim_setup({"stopTime": 10.0})
        # Setup Calibrator
        cal = Calibrator.Calibrator(goals, tuner_para, quality_measure, optimizer_method, dymola_api, **cal_kwargs)
        # Calibrate
        res = cal.calibrate(cal.objective)
        # Right now this only prints the result
        cal.save_result(res, working_dir, ftype="pdf")


if __name__ == "__main__":
    example(continouus=True)
