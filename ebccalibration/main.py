"""Main file for coordination of all steps of a calibration.
E.g. Preprocessing, classifier, calibration etc."""

from ebccalibration.calibration import DymolaAPI
from ebccalibration.calibration import Calibrator
import os

def example():
    """Example function for a calibration process"""
    cwdir = os.path.normpath(r"D:")
    # declaring aliases, goals and tuners
    #Aliases are used to convert the names in modelica into the names used to calculate the objective.
    #The aliases define the names of "meas" and "sim" in each goal-dict.
    aliases = {"sine.y": "sim",
               "trapezoid.y": "trap_meas",
               "pulse.y": "pulse_meas"}
    goals = [{"meas":"trap_meas",
              "sim":"sim",
              "weighting":0.8},
             {"meas": "pulse_meas",
              "sim": "sim",
              "weighting": 0.2}]
    tunerPara = {"amplitude": {"start": 1, "uppBou": 3, "lowBou": 0.3},
                 "freqHz":{"start": 0.5, "uppBou": 0.99, "lowBou": 0.001}}
    # Save the dictionaries to xml--> Just for showing how to workflow will be
    goalXML = os.path.join(cwdir, "goalTest.xml")
    tunerXML = os.path.join(cwdir, "tunerTest.xml")
    Calibrator.save_goals_xml(goals, goalXML)
    Calibrator.save_tuner_xml(tunerPara, tunerXML)
    # Reload them--> Just for showing how to workflow will be
    goals = Calibrator.load_goals_xml(goalXML)
    tunerPara = Calibrator.load_tuner_xml(tunerXML)
    # Setup dymAPI
    exPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "calibration","examples", "ExampleCalibration.mo")
    packages = [os.path.normpath(exPath)]
    dymAPI = DymolaAPI.dymolaInterface(cwdir, packages,
                                       "ExampleCalibration")
    dymAPI.set_simSetup({"stopTime": 100.0})
    # Setup Calibrator
    methods = {"disp": True,
               "ftol": 2.220446049250313e-09,
               "eps": 1e-2
               }
    cal = Calibrator.calibrator(goals, tunerPara, "RMSE", "L-BFGS-B", dymAPI, aliases, **{"methods": methods})
    # Calibrate
    res = cal.calibrate(cal.objective)
    # Right now this only prints the result
    cal.save_result(res)

def main():
    """Main Function for calibration. See the example function for how the process works"""
    cwdir = os.path.normpath(r"D:\03_Python_WorkDir\CalibrationTest")
    # declaring goals and tuners
    goals = [{"meas":
                  "meas.y / K",
              "sim":
                  "sim.y / K",
              "weighting":
                  1}]
    tunerPara = {"hpToCalibrate.VCon": {"start": 0.008, "uppBou": 0.018, "lowBou": 0.002}}
    #Save the dictionaries to xml--> Just for showing how to workflow will be
    goalXML = os.path.join(cwdir, "goalTest.xml")
    tunerXML = os.path.join(cwdir, "tunerTest.xml")
    Calibrator.save_goals_xml(goals, goalXML)
    Calibrator.save_tuner_xml(tunerPara, tunerXML)
    #Reload them--> Just for showing how to workflow will be
    goals = Calibrator.load_goals_xml(goalXML)
    tunerPara = Calibrator.load_tuner_xml(tunerXML)
    #Setup dymAPI
    packages = [
        os.path.normpath(r"D:\04_Git\AixLib_development\AixLib\AixLib\package.mo"),
        os.path.normpath(r"D:\04_Git\Calibration_And_Analysis_Of_Hybrid_Heat_Pump_Systems\CalibrationModules\package.mo")]
    dymAPI = DymolaAPI.dymolaInterface(cwdir, packages, "CalibrationModules.ParametricStudies.HeatPump.HeatPumpAdvanced")
    dymAPI.set_simSetup({"stopTime": 30.0})  # 346929.84
    #Setup Calibrator
    #Define aliases
    aliases = {"T_con_out_meas.y": "meas.y",
               "hpToCalibrate.T_con_out": "sim.y"}
    methods = {"disp":False,
                    "ftol":2.220446049250313e-09,
                    "eps":0.1
                   }
    cal = Calibrator.calibrator(goals, tunerPara, "RMSE", "L-BFGS-B", dymAPI, aliases, **{"methods": methods})
    #Calibrate
    res = cal.calibrate(cal.objective)
    #Right now this only prints the result
    cal.save_result(res)

if __name__=="__main__":
    example()