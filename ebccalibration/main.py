"""Main file for coordination of all steps of a calibration.
E.g. Preprocessing, classifier, calibration etc."""

from ebccalibration.calibration import DymolaAPI
from ebccalibration.calibration import Calibrator
import os
from ebcpython.modelica.tools import manipulate_dsin
def continouusExample():
    """Example function for a continouus calibration process with multiple classes"""
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
    tunerPara = {"amplitude": {"start": 2, "uppBou": 6, "lowBou": 0.3},
                 "freqHz": {"start": 0.5, "uppBou": 0.99, "lowBou": 0.001}}
    continouusData = [{"startTime":0.0,
                      "stopTime":10.0,
                       "type": "Anschalten",
                       "goals":goals,
                       "tunerPara": tunerPara},
                      {"startTime": 10.0,
                       "stopTime": 20.0,
                       "type": "Ausschalten",
                       "goals": goals,
                       "tunerPara": tunerPara}]
    # Setup dymAPI
    exPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "calibration", "examples", "ExampleCalibration.mo")
    packages = [os.path.normpath(exPath)]
    dymAPI = DymolaAPI.dymolaInterface(working_dir, packages,
                                       "ExampleCalibration")
    # Setup Calibrator
    method_options = {"maxiter": 100000,       # Maximal iterations. Abort after maxiter even if no minimum has been achieved.
               "disp": False,           # Show additional infos in console
               "ftol": 2.220446049250313e-09,
               "eps": 0.5
               }
    kwargs = {"method_options": method_options,
              #"tol": 0.95,              # Overall objective function tolerance, e.g. minimize until RMSE < 0.95
              "plotCallback": False,
              "saveFiles": False,
              "continouusCalibration": True}
    # Join all initial names:
    totalInitialNames = Calibrator.join_tunerParas(continouusData)
    # Calibrate
    #manipulate_dsin.eliminate_parameters(r"D:\dsfinal.txt", r"D:\test.txt", [],eliminateAuxiliarParmateres=True)
    calHistory = []
    for c in continouusData:
        #Alter the simulation time
        dymAPI.set_simSetup({"startTime":c["startTime"],
                             "stopTime":c["stopTime"]})
        #Alter tunerParas based on old results
        if len(calHistory)>0:
            tunerPara = Calibrator.alterTunerParas(c["tunerPara"], calHistory)
            # Alter the dsfinal for the new phase
            new_dsfinal = os.path.join("D:\\", "dsfinal.txt")
            manipulate_dsin.eliminate_parameters(os.path.join(calHistory[-1]["cal"].savepathMinResult, "dsfinal.txt"), new_dsfinal, totalInitialNames)
            dymAPI.importInitial(new_dsfinal)
        else:
            tunerPara = c["tunerPara"]
        #Create class with new dymAPI
        cal = Calibrator.calibrator(c["goals"], tunerPara, "RMSE", "L-BFGS-B", dymAPI, **kwargs)
        res = cal.calibrate(cal.objective)
        if hasattr(cal, "trajNames"):
            totalInitialNames = list(set(totalInitialNames + cal.trajNames))
        calHistory.append({"cal":cal,
                           "res": res,
                           "continouusData": c})
    dymAPI.dymola.close()
    #cal.save_result(res, working_dir, ftype="pdf")

def example():
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
    # Setup dymAPI
    exPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "calibration", "examples", "ExampleCalibration.mo")
    packages = [os.path.normpath(exPath)]
    dymAPI = DymolaAPI.dymolaInterface(working_dir, packages,
                                       "ExampleCalibration")
    dymAPI.set_simSetup({"stopTime": 10.0})
    # Setup Calibrator
    method_options = {"maxiter": 100000,       # Maximal iterations. Abort after maxiter even if no minimum has been achieved.
               "disp": False,           # Show additional infos in console
               "ftol": 2.220446049250313e-09,
               "eps": 0.1
               }
    kwargs = {"method_options": method_options,
              #"tol": 0.95,              # Overall objective function tolerance, e.g. minimize until RMSE < 0.95
              "plotCallback": True}
    cal = Calibrator.calibrator(goals, tunerPara, "NRMSE", "L-BFGS-B", dymAPI, **kwargs)
    # Calibrate
    res = cal.calibrate(cal.objective)
    # Right now this only prints the result
    cal.save_result(res, working_dir, ftype="pdf")


if __name__ == "__main__":
    continouusExample()
