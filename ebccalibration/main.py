"""Main file for coordination of all steps of a calibration.
E.g. Preprocessing, classifier, calibration etc."""

from calibration import DymolaAPI
from calibration import Calibrator

def main():
    cwdir = r"D:\testzone"

    dymAPI = DymolaAPI.dymolaInterface(cwdir, packages, modelName)
    cal = Calibrator.calibrator("RMSE", "L-BFSQ-B", bounds, dymAPI)
    res = cal.calibrate(cal.objective, startSet)
    cal.save_result(res)


if __name__=="__main__":
