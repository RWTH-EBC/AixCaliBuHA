"""Test-module for all classes inside
ebcpy.data_types."""

import unittest
from aixcalibuha import CalibrationClass


class TestDataTypes(unittest.TestCase):
    """Test-class for the data_types module of ebcpy."""

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


if __name__ == "__main__":
    unittest.main()
