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

    def test_sa_morris(self):
        """
        Function to test the sensitivity analyzer class using morris
        """
        # Setup the problem
        sen_ana = MorrisAnalyzer(
            sim_api=self.sim_api,
            num_samples=2,
            cd=self.sim_api.cd,
            analysis_variable='mu_star'
        )
        self._run_sen_ana(sen_ana, 'mu_star')

    def test_sa_sobol(self):
        """
        Function to test the sensitivity analyzer class using sobol
        """
        # Setup the problem
        sen_ana = SobolAnalyzer(
            sim_api=self.sim_api,
            num_samples=1,
            cd=self.sim_api.cd,
            analysis_variable='S1'
        )
        self._run_sen_ana(sen_ana, 'S1')

    def _run_sen_ana(self, sen_ana, analysis_variable):
        # Choose initial_values and set boundaries to tuner_parameters
        # Evaluate which tuner_para has influence on what class
        sen_result, classes = sen_ana.run(self.calibration_classes)
        print(sen_ana.__class__.__name__)
        if sen_ana.__class__.__name__ == 'SobolAnalyzer':
            sen_result = sen_result[0]
        self.assertIsInstance(sen_result, pd.DataFrame)
        self.assertIsInstance(classes, list)
        for _cls in classes:
            self.assertIsInstance(_cls, CalibrationClass)
        classes = sen_ana.select_by_threshold(calibration_classes=classes,
                                              result=sen_result,
                                              analysis_variable=analysis_variable,
                                              threshold=0)
        self.assertIsInstance(classes, list)
        self.assertTrue(len(classes) >= 1)
        with self.assertRaises(ValueError):
            sen_ana.select_by_threshold(
                calibration_classes=classes,
                result=sen_result,
                analysis_variable=analysis_variable,
                threshold=np.inf)

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
