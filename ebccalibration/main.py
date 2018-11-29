"""Main file for coordination of all steps of a calibration.
E.g. Preprocessing, classifier, calibration etc."""

from ebccalibration.calibration import DymolaAPI
from ebccalibration.calibration import Calibrator
import os

def main():
    """Main Function for calibration"""
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
    main()