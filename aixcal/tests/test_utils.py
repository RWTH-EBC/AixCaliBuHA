"""Test-module for all classes inside
aixcal.utils."""

import unittest
import os
import numpy as np
from aixcal.utils import visualizer
from aixcal.utils import statistics_analyzer


class TestStatisticsAnalyzer(unittest.TestCase):
    """Test-class for the StatisticsAnalyzer-Class"""

    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        self.framework_dir = os.path.dirname(os.path.dirname(__file__))
        self.example_dir = os.path.normpath(self.framework_dir + "//examples")
        self.meas_ex = np.random.rand(1000)
        self.sim_ex = np.random.rand(1000)*10

    def test_calc(self):
        """Test class StatisticsAnalyzer"""
        sup_methods = statistics_analyzer.StatisticsAnalyzer._supported_methods
        for method in sup_methods:
            stat_analyzer = statistics_analyzer.StatisticsAnalyzer(method)
            self.assertIsInstance(stat_analyzer.calc(self.meas_ex, self.sim_ex),
                                  float)
        with self.assertRaises(ValueError):
            stat_analyzer = statistics_analyzer.StatisticsAnalyzer("not_supported_method")

    def test_calc_RMSE(self):
        """Test static function calc_RMSE"""
        stat_analyzer = statistics_analyzer.StatisticsAnalyzer
        self.assertIsInstance(stat_analyzer.calc_RMSE(self.meas_ex, self.sim_ex),
                              float)

    def test_calc_MSE(self):
        """Test static function calc_MSE"""
        stat_analyzer = statistics_analyzer.StatisticsAnalyzer
        self.assertIsInstance(stat_analyzer.calc_MSE(self.meas_ex, self.sim_ex),
                              float)

    def test_calc_MAE(self):
        """Test static function calc_MAE"""
        stat_analyzer = statistics_analyzer.StatisticsAnalyzer
        self.assertIsInstance(stat_analyzer.calc_MAE(self.meas_ex, self.sim_ex),
                              float)

    def test_calc_NRMSE(self):
        """Test static function calc_NRMSE"""
        stat_analyzer = statistics_analyzer.StatisticsAnalyzer
        self.assertIsInstance(stat_analyzer.calc_NRMSE(self.meas_ex, self.sim_ex),
                              float)

    def test_calc_CVRMSE(self):
        """Test static function calc_CVRMSE"""
        stat_analyzer = statistics_analyzer.StatisticsAnalyzer
        self.assertIsInstance(stat_analyzer.calc_CVRMSE(self.meas_ex, self.sim_ex),
                              float)

    def test_calc_R2(self):
        """Test static function calc_R2"""
        stat_analyzer = statistics_analyzer.StatisticsAnalyzer
        self.assertIsInstance(stat_analyzer.calc_R2(self.meas_ex, self.sim_ex),
                              float)


class TestVisualizer(unittest.TestCase):
    """Test-class for the visualizer module."""

    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        self.framework_dir = os.path.dirname(os.path.dirname(__file__))
        self.example_dir = os.path.normpath(self.framework_dir + "//examples")
        self.logger = visualizer.Visualizer(self.example_dir, "test_logger")

    def test_logging(self):
        """Test if logging works."""
        example_str = "This is a test"
        self.logger.log(example_str)
        with open(self.logger.filepath_log, "r") as logfile:
            logfile.seek(0)
            self.assertEqual(logfile.read()[-len(example_str):], example_str)

    @unittest.skip("Not yet implemented.")
    def test_correct_format(self):
        """Test if the output format for different string works."""
        pass  # TODO Implement test

    def tearDown(self):
        """Remove created files."""
        os.remove(self.logger.filepath_log)


if __name__ == "__main__":
    unittest.main()
