"""Test-module for all classes inside
aixcal.optimization.calibration."""

import unittest
import os
import shutil
from aixcal.optimizer import calibration
from aixcal.simulationapi.dymola_api import DymolaAPI
from aixcal import data_types


class TestModelicaCalibrator(unittest.TestCase):
    """Test-class for the ModelicaCalibrator-class."""

    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        #%% Define relevant paths
        framework_dir = os.path.dirname(os.path.dirname(__file__))
        example_dir = os.path.normpath(framework_dir + "//examples")
        self.example_cal_dir = os.path.normpath(example_dir + "//test_calibration")
        example_mat_file = os.path.normpath(example_dir + "//Modelica//ref_result.mat")

        #%% As all test rely on goals, tuner_parameter and a
        # calibration class, we will always create them here
        tuner_paras = data_types.TunerParas(["heatConv_a", "heatConv_b", "C", "m_flow_2"],
                                            [130, 220, 5000, 0.04],
                                            [(100, 500), (100, 500),
                                             (200, 10000), (0.005, 0.05)])
        mtd = data_types.MeasTargetData(example_mat_file)
        std = data_types.SimTargetData(example_mat_file)
        cols = ["heater.heatPorts[1].T", "heater1.heatPorts[1].T"]
        goals = data_types.Goals(mtd, std, meas_columns=cols,
                                 sim_columns=cols, weightings=[0.7, 0.3])
        self.calibration_class = data_types.CalibrationClass("Device On", 0, 3600,
                                                             goals=goals,
                                                             tuner_paras=tuner_paras)
        self.calibration_classes = [data_types.CalibrationClass("Device On", 0, 1200,
                                                                goals=goals,
                                                                tuner_paras=tuner_paras),
                                    data_types.CalibrationClass("Device Off", 1200, 3600,
                                                                goals=goals,
                                                                tuner_paras=tuner_paras)]

        self.statistical_measure = "NRMSE"
        # %% Instantiate dymola-api
        packages = [os.path.normpath(example_dir + "//Modelica//AixCalTest//package.mo")]
        model_name = "AixCalTest.TestModel"
        try:
            self.dym_api = DymolaAPI(self.example_cal_dir, model_name, packages)
        except (FileNotFoundError, ImportError, ConnectionError):
            self.skipTest("Could not load the dymola interface on this machine.")

    def test_modelica_calibrator(self):
        """Function for testing of class calibration.ModelicaCalibrator."""
        modelica_calibrator = calibration.ModelicaCalibrator(self.example_cal_dir,
                                                             self.dym_api,
                                                             self.statistical_measure,
                                                             self.calibration_class,
                                                             num_function_calls=5)
        # Test run for scipy and L-BFGS-B
        modelica_calibrator.run("L-BFGS-B", "dlib")

    def tearDown(self):
        """Remove all created folders while calibrating."""
        try:
            self.dym_api.close()
        except AttributeError:
            pass
        try:
            shutil.rmtree(self.example_cal_dir)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    unittest.main()
