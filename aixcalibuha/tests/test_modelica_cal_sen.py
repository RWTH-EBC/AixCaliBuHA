"""Test-module for all classes inside
aixcalibuha.optimization.calibration and the class
aixcalibuha.sensanalyzer.sensitivity_analyzer.SenAnalyzer"""

import unittest
import os
import shutil
from ebcpy.simulationapi.dymola_api import DymolaAPI
from ebcpy import data_types
from aixcalibuha.calibration import modelica
from aixcalibuha.sensanalyzer import sensitivity_analyzer
from aixcalibuha import CalibrationClass


class TestModelicaCalibrator(unittest.TestCase):
    """Test-class for the ModelicaCalibrator-class."""

    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        #%% Define relevant paths
        framework_dir = os.path.dirname(os.path.dirname(__file__))
        example_dir = os.path.join(framework_dir, "examples")
        self.example_cal_dir = os.path.join(example_dir, "test_calibration")
        example_mat_file = os.path.join(example_dir, "data", "ref_result.mat")

        #%% As all test rely on goals, tuner_parameter and a
        # calibration class, we will always create them here
        tuner_paras = data_types.TunerParas(["heatConv_a", "heatConv_b", "C", "m_flow_2"],
                                            [130, 220, 5000, 0.04],
                                            [(100, 500), (100, 500),
                                             (200, 10000), (0.005, 0.05)])
        mtd = data_types.MeasTargetData(example_mat_file)
        std = data_types.SimTargetData(example_mat_file)
        cols = ["heater.heatPorts[1].T", "heater1.heatPorts[1].T"]
        goals = data_types.Goals(cols, cols, mtd, sim_target_data=std, weightings=[0.7, 0.3])
        self.calibration_class = CalibrationClass("Device On", 0, 3600,
                                                  goals=goals,
                                                  tuner_paras=tuner_paras)
        self.calibration_classes = [CalibrationClass("Device On", 0, 1200,
                                                     goals=goals,
                                                     tuner_paras=tuner_paras),
                                    CalibrationClass("Device Off", 1200, 3600,
                                                     goals=goals,
                                                     tuner_paras=tuner_paras)]

        self.statistical_measure = "MAE"
        # %% Instantiate dymola-api
        packages = [os.path.join(example_dir, "AixCalTest", "package.mo")]
        model_name = "AixCalTest.TestModel"
        try:
            self.dym_api = DymolaAPI(self.example_cal_dir, model_name, packages)
        except (FileNotFoundError, ImportError, ConnectionError):
            self.skipTest("Could not load the dymola interface on this machine.")

    def test_modelica_calibrator(self):
        """Function for testing of class calibration.ModelicaCalibrator."""
        modelica_calibrator = modelica.ModelicaCalibrator("dlib_minimize",
                                                          self.example_cal_dir,
                                                          self.dym_api,
                                                          self.statistical_measure,
                                                          self.calibration_class,
                                                          num_function_calls=5)
        # Test run for scipy and L-BFGS-B
        modelica_calibrator.calibrate(method=None)

    def test_mutliple_class_calibration(self):
        """Function for testing of class calibration.FixStartContModelicaCal."""
        modelica_calibrator = modelica.MultipleClassCalibrator("dlib_minimize",
                                                               self.example_cal_dir,
                                                               self.dym_api,
                                                               self.statistical_measure,
                                                               self.calibration_classes,
                                                               start_time_method='fixstart',
                                                               reference_start_time=0,
                                                               num_function_calls=5)
        modelica_calibrator.calibrate(method=None)

    def test_sen_ana_run(self):
        """
        Function to test the sensitivity analyzer class by running
        the process through.
        """
        # Setup the problem
        sen_problem = sensitivity_analyzer.SensitivityProblem("morris",
                                                              num_samples=2)

        sen_ana = sensitivity_analyzer.SenAnalyzer(self.dym_api.cd,
                                                   simulation_api=self.dym_api,
                                                   sensitivity_problem=sen_problem,
                                                   calibration_classes=self.calibration_classes,
                                                   statistical_measure=self.statistical_measure)

        # Choose initial_values and set boundaries to tuner_parameters
        # Evaluate which tuner_para has influence on what class
        sen_result = sen_ana.run()
        self.assertIsInstance(sen_result, list)
        self.assertIsInstance(sen_result[0], dict)

        cal_classes = sen_ana.automatic_select(self.calibration_classes,
                                               sen_result,
                                               threshold=1)
        self.assertIsInstance(cal_classes, list)
        self.assertIsInstance(cal_classes[0], CalibrationClass)

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
