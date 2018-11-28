"""Main file for coordination of all steps of a calibration.
E.g. Preprocessing, classifier, calibration etc."""

from calibration import DymolaAPI
from calibration import Calibrator
import os

def main():
    """Main Function for calibration"""
    cwdir = r"D:\01_python_workDir"
    #declaring goals and tuners
    goals = [{"meas": "meas.y / ", "sim": "sim.y / ", "weighting":1}]
    goalXML = os.path.join(cwdir, "goalTest.xml")
    tunerPara = {"f":{"start": 0.2, "uppBou": 0.5, "lowBou": 0.1}}
    tunerXML = os.path.join(cwdir, "tunerTest.xml")
    #test save
    Calibrator.save_goals_xml(goals, goalXML)
    Calibrator.save_tuner_xml(tunerPara, tunerXML)
    #test load
    #goals = Calibrator.load_goals_xml(goalXML)
    tunerPara = Calibrator.load_tuner_xml(tunerXML)
    #Setup dymAPI
    packages = [r"D:\00_dymola_WorkDir\testCalibration.mo"]
    dymAPI = DymolaAPI.dymolaInterface(cwdir, packages, "Unnamed1")
    dymAPI.set_simSetup({"stopTime":100.0})
    #Setup Calibrator
    aliases = {"meas.y":"meas.y",
               "sim.y":"sim.y"}
    cal = Calibrator.calibrator(goals,tunerPara, "RMSE", "L-BFGS-B", dymAPI, aliases)
    #Calibrate
    res = cal.calibrate(cal.objective)
    cal.save_result(res)

if __name__=="__main__":
    main()