"""Test-module for all classes inside aixcalibuha.calibration.py"""

import pathlib
import unittest
import os
from aixcalibuha.calibration import Calibrator


class TestCalibrator(unittest.TestCase):
    """Test-class for the data_types module of ebcpy."""

    def test_init(self):
        """Test the correct setup"""
        cal = Calibrator(cd=os.getcwd(),
                         statistical_measure="NRMSE",
                         sim_api=None)
        self.assertEqual(cal.statistical_measure, "NRMSE")
        self.assertEqual(cal.cd, os.getcwd())
        self.assertIsNone(cal.sim_api)

    def test_abc(self):
        """Test abstract methods"""
        cal = Calibrator(cd=os.getcwd(),
                         statistical_measure="NRMSE",
                         sim_api=None)
        with self.assertRaises(NotImplementedError):
            cal.validate()
        with self.assertRaises(NotImplementedError):
            cal.calibrate(framework=None, method=None)
        with self.assertRaises(NotImplementedError):
            cal.obj(xk=None)

if __name__ == "__main__":
    unittest.main()
