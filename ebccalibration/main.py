"""Main file for coordination of all steps of a calibration.
E.g. Preprocessing, classifier, calibration etc."""
from docutils.nodes import contact

from ebccalibration.calibration import DymolaAPI
from ebccalibration.calibration import Calibrator
import os
from ebcpython.modelica.tools import manipulate_dsin


def continouusCalibration(continouusData, typeOfContinouusCalibration, dymAPI, work_dir, qualMeas, method, cal_kwargs, referenceTime = 0):
    """
    :param continouusData:
    List with dictionaries used for continoous calibration
    :param typeOfContinouusCalibration: str
    Three modes are possible: dsfinal, timedelta, fixStart. See \img\typeOfContinouusCalibration.png for an overview.
    For the last two options, the parameter referenceTime either gives the timedelta before each simulation OR the fixStart time.
    :param dymAPI:
    Dymola API class
    :param work_dir: os.path.normpath
    :param qualMeas: str
    See Calibrator.calibrator
    :param method: str
    See Calibrator.calibrator
    :param cal_kwargs:
    See Calibrator.calibrator
    :param referenceTime: float or int
    Used as a offset value for the simulation time OR as a fixed startTime.
    :return:
    """
    assert typeOfContinouusCalibration in ["dsfinal", "timedelta", "fixStart"], "The given type of continouus calibration %s does not match a given choice."%typeOfContinouusCalibration
    # Join all initial names:
    totalInitialNames = Calibrator.join_tunerParas(continouusData)
    # Calibrate
    # manipulate_dsin.eliminate_parameters(r"D:\dsfinal.txt", r"D:\test.txt", [],eliminateAuxiliarParmateres=True)
    calHistory = []
    curr_num = 0
    for c in continouusData:
        # Alter the simulation time. This depends on the mode one is using.
        if typeOfContinouusCalibration == "timedelta":
            startTime = float(c["startTime"]-referenceTime)
        elif typeOfContinouusCalibration == "fixStart":
            startTime = float(referenceTime)
        elif typeOfContinouusCalibration == "dsfinal":
            startTime = float(c["startTime"])
        dymAPI.set_simSetup({"startTime": startTime,
                             "stopTime": float(c["stopTime"])})
        # Alter the working directory for the simulations
        cwdir_of_class = os.path.join(work_dir, "%s_%s"%(curr_num, c["class"]))
        dymAPI.set_cwdir(cwdir_of_class)
        # Alter tunerParas based on old results
        if len(calHistory) > 0:
            tunerPara = Calibrator.alterTunerParas(c["tunerPara"], calHistory)
            # Alter the dsfinal for the new phase
            if typeOfContinouusCalibration == "dsfinal":
                new_dsfinal = os.path.join(dymAPI.cwdir, "dsfinal.txt")
                manipulate_dsin.eliminate_parameters(
                    os.path.join(calHistory[-1]["cal"].savepathMinResult, "dsfinal.txt"), new_dsfinal,
                    totalInitialNames)
                dymAPI.importInitial(new_dsfinal)
        else:
            tunerPara = c["tunerPara"]
        # Create class with new dymAPI
        print("Starting with class {} in the time period: start = {} to end = {}".format(c["class"], c["startTime"], c["stopTime"]))
        cal = Calibrator.calibrator(goals=c["goals"],
                                    tunerPara=tunerPara,
                                    qualMeas=qualMeas,
                                    method=method,
                                    dymAPI=dymAPI,
                                    timeInfoTupel = (c["startTime"], c["stopTime"]),
                                    **cal_kwargs)
        res = cal.calibrate(cal.objective)
        if hasattr(cal, "trajNames"):
            totalInitialNames = list(set(totalInitialNames + cal.trajNames))
        calHistory.append({"cal": cal,
                           "res": res,
                           "continouusData": c})
        curr_num += 1

    dymAPI.dymola.close()
    print("Final parameter values after calibration:")
    print(Calibrator._get_continouusAverages(calHistory))


def example(continouus = False):
    """Example function for a calibration process"""
    inputPath = input("Please enter the directory to excecute and save the results of this example:")
    working_dir = os.path.normpath(inputPath)
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
    tunerPara = {"amplitude": {"start": 2, "uppBou": 3, "lowBou": 0.3},
                 "freqHz": {"start": 0.5, "uppBou": 0.99, "lowBou": 0.001}}
    # Save the dictionaries to xml--> Just for showing how to workflow will be
    goalXML = os.path.join(working_dir, "goalTest.xml")
    tunerXML = os.path.join(working_dir, "tunerTest.xml")
    Calibrator.saveXML(goalXML,goals)
    Calibrator.saveXML(tunerXML,tunerPara)
    # Reload them--> Just for showing how to workflow will be
    goals = Calibrator.load_goals_xml(goalXML)
    tunerPara = Calibrator.load_tuner_xml(tunerXML)
    if continouus:
        continouusData = [{"startTime": 0.0,
                           "stopTime": 10.0,
                           "class": "Anschalten",
                           "goals": goals,
                           "tunerPara": tunerPara},
                          {"startTime": 10.0,
                           "stopTime": 20.0,
                           "class": "Ausschalten",
                           "goals": goals,
                           "tunerPara": tunerPara}]
    # Setup dymAPI
    exPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "calibration", "examples", "ExampleCalibration.mo")
    packages = [os.path.normpath(exPath)]
    dymAPI = DymolaAPI.dymolaInterface(working_dir, packages,
                                       "ExampleCalibration")
    method_options = {"maxiter": 100000,       # Maximal iterations. Abort after maxiter even if no minimum has been achieved.
               "disp": False,           # Show additional infos in console
               "ftol": 2.220446049250313e-09,
               "eps": 0.1
               }
    cal_kwargs = {"method_options": method_options,
              #"tol": 0.95,              # Overall objective function tolerance, e.g. minimize until RMSE < 0.95
              "plotCallback": True,
              "saveFiles": False,
              "continouusCalibration": continouus}
    quality_measure = "NRMSE"
    optimizer_method = "L-BFGS-B"

    if continouus:
        continouusCalibration(continouusData=continouusData,
                              typeOfContinouusCalibration = "timedelta",
                              dymAPI=dymAPI,
                              work_dir=working_dir,
                              qualMeas=quality_measure,
                              method=optimizer_method,
                              cal_kwargs=cal_kwargs,
                              referenceTime=1000)
    else:
        dymAPI.set_simSetup({"stopTime": 10.0})
        # Setup Calibrator
        cal = Calibrator.calibrator(goals, tunerPara, quality_measure, optimizer_method, dymAPI, **cal_kwargs)
        # Calibrate
        res = cal.calibrate(cal.objective)
        # Right now this only prints the result
        cal.save_result(res, working_dir, ftype="pdf")


if __name__ == "__main__":
    example(continouus=True)
