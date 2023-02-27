"""Test-module for all classes inside
aixcalibuha.optimization.calibration and the class
aixcalibuha.sensitivity_analysis.sensitivity_analyzer.SenAnalyzer"""

import unittest
import sys
import pathlib
import shutil
import numpy as np
import pandas as pd
from ebcpy import FMU_API, TimeSeriesData
from aixcalibuha import MorrisAnalyzer, SobolAnalyzer, MultipleClassCalibrator, \
    Calibrator, CalibrationClass, TunerParas, Goals


class TestModelicaCalibrator(unittest.TestCase):
    """Test-class for the Calibrator-class."""

    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        #%% Define relevant paths
        aixcalibuha_dir = pathlib.Path(__file__).parents[1]
        self.data_dir = aixcalibuha_dir.joinpath("examples", "data")
        self.example_cal_dir = aixcalibuha_dir.joinpath("tests", "testzone")

        # As the examples should work, and the cal_class example uses the other examples,
        # we will test it here:
        meas_target_data = TimeSeriesData(
            self.data_dir.joinpath("PumpAndValve.hdf"), key="examples"
        )
        meas_target_data.to_float_index()

        # Setup three variables for different format of setup
        variable_names = {
            "TCap": ["TCapacity", "heatCapacitor.T"],
            "TPipe": {"meas": "TPipe", "sim": "pipe.T"}
        }

        tuner_paras = TunerParas(names=["speedRamp.duration", "valveRamp.duration"],
                                 initial_values=[0.1, 0.1],
                                 bounds=[(0.1, 10), (0.1, 10)])
        # Real "best" values: speedRamp.duration=0.432 and valveRamp.duration=2.5423
        # Check setup the goals class:
        goals = Goals(meas_target_data=meas_target_data,
                      variable_names=variable_names,
                      statistical_measure="NRMSE")
        self.calibration_classes = [
            CalibrationClass(name="First", start_time=0, stop_time=1,
                             goals=goals, tuner_paras=tuner_paras),
            CalibrationClass(name="Second", start_time=1, stop_time=10,
                             goals=goals, tuner_paras=tuner_paras)
        ]

        # %% Instantiate dymola-api
        if "win" in sys.platform:
            model_name = aixcalibuha_dir.joinpath("examples", "model", "PumpAndValve_windows.fmu")
        else:
            model_name = aixcalibuha_dir.joinpath("examples", "model", "PumpAndValve_linux.fmu")

        self.sim_api = FMU_API(cd=self.example_cal_dir,
                               model_name=model_name)

    def test_modelica_calibrator(self):
        """Function for testing of class calibration.Calibrator."""
        calibrator = Calibrator(
            cd=self.sim_api.cd,
            sim_api=self.sim_api,
            calibration_class=self.calibration_classes[0],
            show_plot=False,
            max_itercount=5)
        # Test run for scipy and L-BFGS-B
        calibrator.calibrate(framework="scipy_differential_evolution",
                             method="best1bin")

    def test_mutliple_class_calibration(self):
        """Function for testing of class calibration.FixStartContModelicaCal."""
        calibrator = MultipleClassCalibrator(
            cd=self.example_cal_dir,
            sim_api=self.sim_api,
            calibration_classes=self.calibration_classes,
            start_time_method='fixstart',
            fix_start_time=0,
            show_plot=False,
            max_itercount=5)

        calibrator.calibrate(framework="scipy_differential_evolution",
                             method="best1bin")

    def tearDown(self):
        """Remove all created folders while calibrating."""
        try:
            self.sim_api.close()
        except AttributeError:
            pass
        try:
            shutil.rmtree(self.example_cal_dir, ignore_errors=True)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    unittest.main()
