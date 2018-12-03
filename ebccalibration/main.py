"""Main file for coordination of all steps of a calibration.
E.g. Preprocessing, classifier, calibration etc."""

from ebccalibration.calibration import DymolaAPI
from ebccalibration.calibration import Calibrator
import os


def example():
    """Example function for a calibration process"""
    working_dir = os.path.normpath(r"D:\03_Python_WorkDir\00_Testzone")
    # Declaring aliases, goals and tuners:
    # Aliases are used to convert the names in modelica into the names used to calculate the objective.
    # The aliases define the names of "meas" and "sim" in each goal-dict.
    # Measurement values must be passed through the Modelica simulation.
    aliases = {"sine.y": "sim",
               "trapezoid.y": "trap_meas",
               "pulse.y": "pulse_meas"}
    goals = [{"meas": "trap_meas",
              "sim": "sim",
              "weighting": 0.8},
             {"meas": "pulse_meas",
              "sim": "sim",
              "weighting": 0.2}]
    tunerPara = {"amplitude": {"start": 0.3, "uppBou": 3, "lowBou": 0.3},
                 "freqHz": {"start": 0.001, "uppBou": 0.99, "lowBou": 0.001}}
    # Save the dictionaries to xml--> Just for showing how to workflow will be
    goalXML = os.path.join(working_dir, "goalTest.xml")
    tunerXML = os.path.join(working_dir, "tunerTest.xml")
    Calibrator.save_goals_xml(goals, goalXML)
    Calibrator.save_tuner_xml(tunerPara, tunerXML)
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
               "eps": 0.001
               }
    kwargs = {"method_options": method_options,
              "tol": 0.95,              # Overall objective function tolerance, e.g. minimize until RMSE < 0.95
              "plotCallback": False}
    cal = Calibrator.calibrator(goals, tunerPara, "RMSE", "L-BFGS-B", dymAPI, aliases, **kwargs)
    # Calibrate
    res = cal.calibrate(cal.objective)
    # Right now this only prints the result
    cal.save_result(res, working_dir, ftype="pdf")


if __name__ == "__main__":
    example()
