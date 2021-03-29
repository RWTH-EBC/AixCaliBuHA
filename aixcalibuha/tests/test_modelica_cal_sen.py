"""Test-module for all classes inside
aixcalibuha.optimization.calibration and the class
aixcalibuha.sensanalyzer.sensitivity_analyzer.SenAnalyzer"""

import unittest
import os
import shutil
from ebcpy.simulationapi.dymola_api import DymolaAPI
from aixcalibuha.calibration import modelica
from aixcalibuha.sensanalyzer import MorrisAnalyzer, SobolAnalyzer
from aixcalibuha import CalibrationClass
from aixcalibuha.examples import cal_classes_example


class TestModelicaCalibrator(unittest.TestCase):
    """Test-class for the ModelicaCalibrator-class."""

    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        #%% Define relevant paths
        framework_dir = os.path.dirname(os.path.dirname(__file__))
        example_dir = os.path.join(framework_dir, "examples")
        self.example_cal_dir = os.path.join(example_dir, "test_calibration")

        # As the examples should work, and the cal_class example uses the other examples,
        # we will test it here:
        self.calibration_classes = cal_classes_example.setup_calibration_classes()

        self.statistical_measure = "NRMSE"
        # %% Instantiate dymola-api
        packages = [os.path.join(example_dir, "AixCalTest", "package.mo")]
        model_name = "AixCalTest.TestModel"
        try:
            self.dym_api = DymolaAPI(self.example_cal_dir, model_name, packages)
        except (FileNotFoundError, ImportError, ConnectionError):
            self.skipTest("Could not load the dymola interface on this machine.")
        try:
            import dlib
        except ImportError:
            self.skipTest("Tests only work with dlib installed.")

    def test_modelica_calibrator(self):
        """Function for testing of class calibration.ModelicaCalibrator."""
        modelica_calibrator = modelica.ModelicaCalibrator(self.example_cal_dir,
                                                          self.dym_api,
                                                          self.statistical_measure,
                                                          self.calibration_classes[0],
                                                          num_function_calls=5,
                                                          show_plot=False)
        # Test run for scipy and L-BFGS-B
        modelica_calibrator.calibrate(framework="dlib_minimize", method=None)

    def test_mutliple_class_calibration(self):
        """Function for testing of class calibration.FixStartContModelicaCal."""
        modelica_calibrator = modelica.MultipleClassCalibrator(self.example_cal_dir,
                                                               self.dym_api,
                                                               self.statistical_measure,
                                                               self.calibration_classes,
                                                               start_time_method='fixstart',
                                                               fix_start_time=0,
                                                               num_function_calls=5,
                                                               show_plot=False)

        modelica_calibrator.calibrate(framework="dlib_minimize", method=None)

    def test_sa_morris(self):
        """
        Function to test the sensitivity analyzer class using morris
        """
        # Setup the problem
        sen_ana = MorrisAnalyzer(
            sim_api=self.dym_api,
            statistical_measure=self.statistical_measure,
            num_samples=1,
            cd=self.dym_api.cd,
            analysis_variable='mu_star'
        )
        self._run_sen_ana(sen_ana)

    def test_sa_sobol(self):
        """
        Function to test the sensitivity analyzer class using sobol
        """
        # Setup the problem
        sen_ana = SobolAnalyzer(
            sim_api=self.dym_api,
            statistical_measure=self.statistical_measure,
            num_samples=1,
            cd=self.dym_api.cd,
            analysis_variable='S1'
        )
        self._run_sen_ana(sen_ana)

    def _run_sen_ana(self, sen_ana):
        # Choose initial_values and set boundaries to tuner_parameters
        # Evaluate which tuner_para has influence on what class
        sen_result, classes = sen_ana.run(self.calibration_classes)
        self.assertIsInstance(sen_result, list)
        self.assertIsInstance(sen_result[0], dict)

        # Test automatic run:
        cal_classes = sen_ana.automatic_run(self.calibration_classes)
        self.assertIsInstance(cal_classes, list)
        self.assertIsInstance(cal_classes[0], CalibrationClass)

    def tearDown(self):
        """Remove all created folders while calibrating."""
        try:
            self.dym_api.close()
        except AttributeError:
            pass
        try:
            shutil.rmtree(self.example_cal_dir, ignore_errors=True)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    unittest.main()
