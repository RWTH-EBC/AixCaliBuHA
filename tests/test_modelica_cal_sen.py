"""Test-module for all classes inside
aixcalibuha.optimization.calibration and the class
aixcalibuha.sensanalyzer.sensitivity_analyzer.SenAnalyzer"""

import unittest
import os
import pathlib
import shutil
from ebcpy.simulationapi.dymola_api import DymolaAPI
from aixcalibuha.calibration import MultipleClassCalibrator, Calibrator
from aixcalibuha.sensanalyzer import MorrisAnalyzer, SobolAnalyzer
from aixcalibuha import CalibrationClass
from aixcalibuha.examples import cal_classes_example


class TestModelicaCalibrator(unittest.TestCase):
    """Test-class for the Calibrator-class."""

    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        #%% Define relevant paths
        framework_dir = pathlib.Path(__file__).parents[1]
        example_dir = os.path.join(framework_dir, "aixcalibuha", "examples")
        self.example_cal_dir = os.path.join(example_dir, "test_calibration")

        # As the examples should work, and the cal_class example uses the other examples,
        # we will test it here:
        self.calibration_classes = cal_classes_example.setup_calibration_classes()

        for cal_class in self.calibration_classes:
            cal_class.goals.statistical_measure = "NRMSE"
        # %% Instantiate dymola-api
        packages = [os.path.join(example_dir, "AixCalTest", "package.mo")]
        model_name = "AixCalTest.TestModel"
        try:
            self.dym_api = DymolaAPI(self.example_cal_dir, model_name, packages)
        except (FileNotFoundError, ImportError, ConnectionError) as err:
            self.skipTest(f"Could not load the dymola "
                          f"interface on this machine: {err}")
        try:
            import dlib
        except ImportError:
            self.skipTest("Tests only work with dlib installed.")

    def test_modelica_calibrator(self):
        """Function for testing of class calibration.Calibrator."""
        try:
            import dlib
        except ImportError:
            self.skipTest("Tests only work with dlib installed.")
        calibrator = Calibrator(self.example_cal_dir,
                                self.dym_api,
                                self.calibration_classes[0],
                                show_plot=False)
        # Test run for scipy and L-BFGS-B
        calibrator.calibrate(framework="dlib_minimize",
                             method=None,
                             num_function_calls=5)

    def test_mutliple_class_calibration(self):
        """Function for testing of class calibration.FixStartContModelicaCal."""
        calibrator = MultipleClassCalibrator(self.example_cal_dir,
                                             self.dym_api,
                                             self.calibration_classes,
                                             start_time_method='fixstart',
                                             fix_start_time=0,
                                             show_plot=False)

        calibrator.calibrate(framework="dlib_minimize",
                             method=None,
                             num_function_calls=5)

    def test_sa_morris(self):
        """
        Function to test the sensitivity analyzer class using morris
        """
        # Setup the problem
        sen_ana = MorrisAnalyzer(
            sim_api=self.dym_api,
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
