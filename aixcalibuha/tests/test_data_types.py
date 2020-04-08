"""Test-module for all classes inside
ebcpy.data_types."""

import os
import unittest
import numpy as np
from ebcpy import data_types
from aixcalibuha import CalibrationClass, Goals


class TestDataTypes(unittest.TestCase):
    """Test-class for the data_types module of ebcpy."""

    def setUp(self):
        """Called before every test.
        Define example paths and parameters used in all test-functions.
        """
        self.framework_dir = os.path.dirname(os.path.dirname(__file__))
        self.example_dir = os.path.join(self.framework_dir, "examples", "data")
        self.example_data_hdf_path = os.path.join(self.example_dir, "example_data.hdf")

    def test_calibration_class(self):
        """Test the class CalibrationClass"""
        with self.assertRaises(ValueError):
            # Test if start-time higher than stop-time raises an error.
            CalibrationClass("dummy", 100, 50)
        with self.assertRaises(TypeError):
            # Test if a given name not equal to string raises an error.
            not_a_string = 1
            CalibrationClass(not_a_string, 0, 10)

        # Test set_functions for goals and tuner parameters
        dummy_tuner_para = "not TunerParas-Class"
        dummy_goal = "not Goals-Class"
        dummy_cal_class = CalibrationClass("dummy", 0, 10)
        with self.assertRaises(TypeError):
            dummy_cal_class.set_tuner_paras(dummy_tuner_para)
        with self.assertRaises(TypeError):
            dummy_cal_class.set_goals(dummy_goal)


    def test_goals(self):
        """Test the class Goals"""
        # Define some data.
        sim_target_data = data_types.TimeSeriesData(self.example_data_hdf_path, key="parameters")
        meas_target_data = data_types.TimeSeriesData(self.example_data_hdf_path, key="parameters")

        # Setup three variables for different format of setup
        var_names = {"Var_1": ["sine.amplitude / ", "sine.freqHz / Hz"],
                     "Var_2": ("sine.phase / rad", "sine.freqHz / Hz"),
                     "Var_3": {"meas": "sine.startTime / s", "sim": "sine.freqHz / Hz"}}

        # Check setup the goals class:
        goals = Goals(meas_target_data=meas_target_data,
                      variable_names=var_names)

        # Check set_sim_target_data:
        goals.set_sim_target_data(sim_target_data)

        # Check the eval_difference function:
        self.assertIsInstance(goals.eval_difference("RMSE"), float)
        # Try to alter the sim_target_data object with something wrong
        with self.assertRaises(TypeError):
            goals.set_sim_target_data([])
        # Play around with wrong weightings:
        with self.assertRaises(IndexError):
            weightings = [1, 2, 4, 5, 6]
            Goals(meas_target_data=meas_target_data,
                  variable_names=var_names,
                  weightings=weightings)
        with self.assertRaises(IndexError):
            weightings = np.ones(100)/100
            Goals(meas_target_data=meas_target_data,
                  variable_names=var_names,
                  weightings=weightings)


if __name__ == "__main__":
    unittest.main()
