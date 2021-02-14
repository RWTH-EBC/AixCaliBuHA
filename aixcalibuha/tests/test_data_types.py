"""Test-module for all classes inside aixcalibuha.__init__.py"""

import os
import unittest
import numpy as np
from ebcpy import data_types
from aixcalibuha import CalibrationClass, Goals, TunerParas


class TestDataTypes(unittest.TestCase):
    """Test-class for the data_types module of ebcpy."""

    def setUp(self):
        """Called before every test.
        Define example paths and parameters used in all test-functions.
        """
        self.framework_dir = os.path.dirname(os.path.dirname(__file__))
        self.example_dir = os.path.join(self.framework_dir, "examples", "data")
        self.example_data_hdf_path = os.path.join(self.example_dir, "ref_result.hdf")

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
            dummy_cal_class.tuner_paras = dummy_tuner_para
        with self.assertRaises(TypeError):
            dummy_cal_class.goals = dummy_goal

    def test_goals(self):
        """Test the class Goals"""
        # Define some data.
        sim_target_data = data_types.TimeSeriesData(os.path.join(self.example_dir,
                                                                 "simTargetData.mat"))

        meas_target_data = data_types.TimeSeriesData(self.example_data_hdf_path, key="FloatIndex")

        # Setup three variables for different format of setup
        var_names = {"Var_1": ["measured_T_heater_1", "heater1.heatPorts[1].T"],
                     "Var_2": {"meas": "measured_T_heater", "sim": "heater.heatPorts[1].T"}}

        # Check setup the goals class:
        goals = Goals(meas_target_data=meas_target_data,
                      variable_names=var_names)

        # Check set_sim_target_data:
        goals.set_sim_target_data(sim_target_data)

        # Set relevant time interval test:
        goals.set_relevant_time_intervals([(0, 100)])

        # Check the eval_difference function:
        self.assertIsInstance(goals.eval_difference("RMSE"), float)
        # Try to alter the sim_target_data object with something wrong
        with self.assertRaises(IndexError):
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

    def test_tuner_paras(self):
        """Test the class TunerParas"""
        dim = np.random.randint(1, 100)
        names = ["test_%s" % i for i in range(dim)]
        initial_values = np.random.rand(dim) * 10  # Values between 0 and 10.
        # Values between -100 and 110
        bounds = [(float(np.random.rand(1))*-100,
                   float(np.random.rand(1))*100 + 10) for i in range(dim)]
        # Check for false input
        with self.assertRaises(ValueError):
            wrong_bounds = [(0, 100),
                            (100, 0)]
            tuner_paras = TunerParas(names,
                                     initial_values,
                                     wrong_bounds)
        with self.assertRaises(ValueError):
            wrong_bounds = [(0, 100) for i in range(dim+1)]
            tuner_paras = TunerParas(names,
                                     initial_values,
                                     wrong_bounds)
        with self.assertRaises(ValueError):
            wrong_initial_values = np.random.rand(100)
            tuner_paras = TunerParas(names,
                                     wrong_initial_values)
        with self.assertRaises(TypeError):
            wrong_names = ["test_0", 123]
            tuner_paras = TunerParas(wrong_names,
                                     initial_values)
        with self.assertRaises(TypeError):
            wrong_initial_values = ["not an int", 123, 123]
            tuner_paras = TunerParas(names,
                                     wrong_initial_values)

        # Check return values of functions:
        tuner_paras = TunerParas(names,
                                 initial_values,
                                 bounds)
        scaled = np.random.rand(dim)  # between 0 and 1
        # Descale and scale again to check if the output is the almost (numeric precision) same
        descaled = tuner_paras.descale(scaled)
        scaled_return = tuner_paras.scale(descaled)
        np.testing.assert_almost_equal(scaled, scaled_return)
        self.assertEqual(names, tuner_paras.get_names())
        np.testing.assert_equal(tuner_paras.get_initial_values(),
                                initial_values)

        tuner_paras.get_bounds()
        val = tuner_paras.get_value("test_0", "min")
        tuner_paras.set_value("test_0", "min", val)
        with self.assertRaises(ValueError):
            tuner_paras.set_value("test_0", "min", 10000)
        with self.assertRaises(ValueError):
            tuner_paras.set_value("test_0", "min", "not_an_int_or_float")
        with self.assertRaises(KeyError):
            tuner_paras.set_value("test_0", "not_a_key", val)
        # Delete a name and check if the name is really gone.
        tuner_paras.remove_names(["test_0"])
        with self.assertRaises(KeyError):
            tuner_paras.get_value("test_0", "min")


if __name__ == "__main__":
    unittest.main()
