"""Test-module for all classes inside
aixcalibuha.optimization.calibration and the class
aixcalibuha.sensitivity_analysis.sensitivity_analyzer.SenAnalyzer"""

import unittest
import sys
import os
import pathlib
import shutil
import numpy as np
import pandas as pd
from ebcpy import FMU_API, TimeSeriesData
from aixcalibuha import MorrisAnalyzer, SobolAnalyzer, FASTAnalyzer, PAWNAnalyzer, \
    MultipleClassCalibrator, Calibrator, CalibrationClass, TunerParas, Goals


def _set_up():
    aixcalibuha_dir = pathlib.Path(__file__).parents[1]
    data_dir = aixcalibuha_dir.joinpath("examples", "data")
    testzone_dir = aixcalibuha_dir.joinpath("tests", "testzone")

    # As the examples should work, and the cal_class example uses the other examples,
    # we will test it here:
    meas_target_data = TimeSeriesData(
        data_dir.joinpath("PumpAndValve.hdf"), key="examples"
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

    return aixcalibuha_dir, data_dir, testzone_dir, tuner_paras, goals


class TestModelicaCalibrator(unittest.TestCase):
    """Test-class for the Calibrator-class."""

    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        # %% Define relevant paths
        aixcalibuha_dir, self.data_dir, self.example_cal_dir, tuner_paras, goals = _set_up()
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


class TestSenAnalyzer(unittest.TestCase):
    """Test-class for the SenAnalyzer classes."""

    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        # Define relevant paths
        aixcalibuha_dir, self.data_dir, self.result_dir, tuner_paras, goals = _set_up()
        self.calibration_classes = [
            CalibrationClass(name="global", start_time=0, stop_time=10,
                             goals=goals, tuner_paras=tuner_paras),
            CalibrationClass(name="heat up", start_time=0, stop_time=1,
                             goals=goals, tuner_paras=tuner_paras),
            CalibrationClass(name="cool down", start_time=1, stop_time=2,
                             goals=goals, tuner_paras=tuner_paras),
            CalibrationClass(name="stationary", start_time=2, stop_time=10,
                             goals=goals, tuner_paras=tuner_paras)
        ]

        # %% Instantiate dymola-api
        if "win" in sys.platform:
            model_name = aixcalibuha_dir.joinpath("examples", "model", "PumpAndValve_windows.fmu")
        else:
            model_name = aixcalibuha_dir.joinpath("examples", "model", "PumpAndValve_linux.fmu")

        self.sim_api = FMU_API(cd=self.result_dir,
                               model_name=model_name)

    def test_sa_morris(self):
        """
        Function to test the sensitivity analyzer class using morris
        Used to test all parent functions
        """
        # Setup the problem
        sen_ana = MorrisAnalyzer(
            sim_api=self.sim_api,
            num_samples=2,
            cd=self.sim_api.cd.__str__()
        )
        sen_result, classes = sen_ana.run(self.calibration_classes,
                                          plot_result=False)
        self._check_sen_run_return(sen_ana, sen_result, classes)
        with self.assertRaises(ValueError):
            sen_ana.run(self.calibration_classes,
                        plot_result=False,
                        verbose=True,
                        use_first_sim=True,
                        n_cpu=9999999999)
        with self.assertRaises(AttributeError):
            sen_ana.run(self.calibration_classes,
                        plot_result=False,
                        verbose=True,
                        use_first_sim=True,
                        n_cpu=2)

        sen_ana = MorrisAnalyzer(
            sim_api=self.sim_api,
            num_samples=2,
            cd=self.sim_api.cd,
            save_files=True,
        )
        false_calc_classes = self.calibration_classes.copy()
        false_calc_classes.append(CalibrationClass(
            name="False", start_time=1, stop_time=11,
            goals=self.calibration_classes[0].goals,
            tuner_paras=self.calibration_classes[0].tuner_paras
        ))
        with self.assertRaises(ValueError):
            sen_ana.run(false_calc_classes,
                        plot_result=False,
                        verbose=True,
                        use_first_sim=True,
                        n_cpu=1)
        sen_ana.run(self.calibration_classes,
                    plot_result=False,
                    verbose=True,
                    use_first_sim=True,
                    n_cpu=1)
        self._check_sen_run_return(sen_ana, sen_result, classes)
        sen_ana.run(self.calibration_classes,
                    plot_result=False,
                    verbose=True,
                    use_first_sim=True,
                    n_cpu=2)
        self._check_sen_run_return(sen_ana, sen_result, classes)
        os.getlogin = lambda: "test_login"
        sen_ana.save_for_reproduction(
            title="SenAnalyzerTest",
            path=self.result_dir,
            log_message="This is just an example",
            remove_saved_files=False,
            exclude_sim_files=True
        )

    def test_sa_sobol(self):
        """
        Function to test the sensitivity analyzer class using sobol
        """
        sen_ana = SobolAnalyzer(
            sim_api=self.sim_api,
            num_samples=1,
            cd=self.sim_api.cd
        )
        sen_result, classes = sen_ana.run(self.calibration_classes,
                                          plot_result=False)
        self._check_sen_run_return(sen_ana, sen_result, classes)

    def test_sa_fast(self):
        """
        Function to test the sensitivity analyzer class using sobol
        """
        sen_ana = FASTAnalyzer(
            sim_api=self.sim_api,
            num_samples=8,
            cd=self.sim_api.cd,
            M=1
        )
        sen_result, classes = sen_ana.run(self.calibration_classes[0],
                                          plot_result=False)
        self._check_sen_run_return(sen_ana, sen_result, classes)

    def test_sa_pawn(self):
        """
        Function to test the sensitivity analyzer class using sobol
        """
        sen_ana = PAWNAnalyzer(
            sim_api=self.sim_api,
            num_samples=2,
            cd=self.sim_api.cd,
            sampler="morris"
        )
        sen_result, classes = sen_ana.run(self.calibration_classes[0],
                                          plot_result=False)
        self._check_sen_run_return(sen_ana, sen_result, classes)
        sen_ana = PAWNAnalyzer(
            sim_api=self.sim_api,
            num_samples=1,
            cd=self.sim_api.cd,
            sampler="sobol"
        )
        sen_result, classes = sen_ana.run(self.calibration_classes[0],
                                          plot_result=False)
        self._check_sen_run_return(sen_ana, sen_result, classes)
        sen_ana = PAWNAnalyzer(
            sim_api=self.sim_api,
            num_samples=8,
            cd=self.sim_api.cd,
            M=1,
            sampler="fast"
        )
        sen_result, classes = sen_ana.run(self.calibration_classes[0],
                                          plot_result=False)
        self._check_sen_run_return(sen_ana, sen_result, classes)

    def test_select_by_threshold(self):
        """
        Function to test the function select_by_threshold of the SenAnalyzer
        """
        sen_result = MorrisAnalyzer.load_from_csv(
            self.data_dir.joinpath("MorrisAnalyzer_results_B.csv"))
        classes = MorrisAnalyzer.select_by_threshold(calibration_classes=self.calibration_classes,
                                                     result=sen_result,
                                                     analysis_variable='mu_star',
                                                     threshold=0)
        self.assertIsInstance(classes, list)
        self.assertTrue(len(classes) >= 1)
        with self.assertRaises(ValueError):
            MorrisAnalyzer.select_by_threshold(
                calibration_classes=classes,
                result=sen_result,
                analysis_variable='mu_star',
                threshold=np.inf)

    def test_select_by_threshold_verbose(self):
        """
        Function to test the function select_by_threshold of the SenAnalyzer
        """
        sen_result = SobolAnalyzer.load_from_csv(
            self.data_dir.joinpath("SobolAnalyzer_results_B.csv"))
        cal_class = SobolAnalyzer.select_by_threshold_verbose(
            calibration_class=self.calibration_classes[0],
            result=sen_result,
            analysis_variable="S1",
            threshold=0,
            calc_names_for_selection=["heat up", "cool down"]
        )
        self.assertIsInstance(cal_class, CalibrationClass)
        with self.assertRaises(NameError):
            cal_class = SobolAnalyzer.select_by_threshold_verbose(
                calibration_class=self.calibration_classes[0],
                result=sen_result,
                analysis_variable="S1",
                threshold=0,
                calc_names_for_selection=["On", "Off"]
            )
        with self.assertRaises(ValueError):
            SobolAnalyzer.select_by_threshold_verbose(
                calibration_class=self.calibration_classes[0],
                result=sen_result,
                analysis_variable="S1",
                threshold=np.inf,
                calc_names_for_selection=["heat up", "cool down"]
            )
        with self.assertRaises(NameError):
            false_cal_class = self.calibration_classes[0]
            false_cal_class.tuner_paras = TunerParas(names=["false.name", "valveRamp.duration"],
                                                     initial_values=[0.1, 0.1],
                                                     bounds=[(0.1, 10), (0.1, 10)])
            SobolAnalyzer.select_by_threshold_verbose(
                calibration_class=self.calibration_classes[0],
                result=sen_result,
                analysis_variable="S1",
                threshold=np.inf,
                calc_names_for_selection=["heat up", "cool down"]
            )

    def test_plot_single(self):
        """
        Function to test the plot function of the SenAnalyzer
        """
        sen_result = MorrisAnalyzer.load_from_csv(
            self.data_dir.joinpath("MorrisAnalyzer_results_B.csv"))
        MorrisAnalyzer.plot_single(result=sen_result,
                                   show_plot=False,
                                   use_suffix=True)

    def test_plot_sobol_s2(self):
        """
        Function to test the second order plot function of the SobolAnalyzer
        """
        sen_result = SobolAnalyzer.load_second_order_from_csv(
            self.data_dir.joinpath("SobolAnalyzer_results_second_order_A.csv"))
        SobolAnalyzer.plot_second_order(sen_result,
                                        use_suffix=True,
                                        show_plot=False)
        SobolAnalyzer.plot_single_second_order(sen_result, "rad.n",
                                               show_plot=False)
        SobolAnalyzer.heatmaps(sen_result,
                               show_plot=False)

    def _check_sen_run_return(self, sen_ana, sen_result, classes):
        if sen_ana.__class__.__name__ == 'SobolAnalyzer':
            self.assertIsInstance(sen_result, tuple)
            self.assertIsInstance(sen_result[1], pd.DataFrame)
            sen_result = sen_result[0]
        self.assertIsInstance(sen_result, pd.DataFrame)
        self.assertIsInstance(classes, list)
        for _cls in classes:
            self.assertIsInstance(_cls, CalibrationClass)

    def tearDown(self):
        """Remove all created folders while calibrating."""
        try:
            self.sim_api.close()
        except AttributeError:
            pass
        try:
            shutil.rmtree(self.result_dir, ignore_errors=True)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    unittest.main()
