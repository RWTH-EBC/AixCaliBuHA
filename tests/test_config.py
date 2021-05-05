"""Test-module for all classes inside aixcalibuha.calibration.py"""

import unittest
import pathlib
import aixcalibuha
from aixcalibuha.utils import configuration


class TestConfiguration(unittest.TestCase):
    """Test-class for the data_types module of ebcpy."""

    def test_import(self):
        """Test the correct impotz"""
        self.assertIsInstance(configuration.default_config, dict)

    def test_funcs(self):
        """Test functions"""
        example_hdf = pathlib.Path(__file__).parents[1].joinpath("aixcalibuha",
                                                                 "examples",
                                                                 "data",
                                                                 "ref_result.hdf")
        var_names = {"Var_1": ["measured_T_heater_1", "heater1.heatPorts[1].T"],
                     "Var_2": {"meas": "measured_T_heater", "sim": "heater.heatPorts[1].T"}}

        cal_class = configuration.get_calibration_classes_from_config(
            configs=[{"tuner_paras": {"names": ["test"],
                                      "initial_values": [10],
                                      "bounds": [(0, 20)]},
                      "goals": {"variable_names": var_names,
                                "meas_target_data": {"data": example_hdf,
                                                     "key": "test"}},
                      "name": "test",
                      "start_time": 0,
                      "stop_time": 10}]
        )[0]

        self.assertIsInstance(cal_class, aixcalibuha.CalibrationClass)
        self.assertIsInstance(cal_class.tuner_paras, aixcalibuha.TunerParas)
        self.assertIsInstance(cal_class.goals, aixcalibuha.Goals)


if __name__ == "__main__":
    unittest.main()
