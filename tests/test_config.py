"""Test-module for all classes inside aixcalibuha.calibration"""

import unittest
from pathlib import Path
import aixcalibuha
from aixcalibuha.utils import configuration


class TestConfiguration(unittest.TestCase):
    """Test-class for the data_types module of ebcpy."""

    def test_import(self):
        """Test the correct impotz"""
        self.assertIsInstance(configuration.default_config, dict)

    def test_funcs(self):
        """Test functions"""
        example_hdf = Path(__file__).parents[1].joinpath("examples",
                                                                 "data",
                                                                 "PumpAndValve.hdf")
        var_names = {"TCap": ["TCapacity", "heatCapacitor.T"],
                     "TPipe": {"meas": "TPipe", "sim": "pipe.T"}}

        cal_class = configuration.get_calibration_classes_from_config(
            config=[{"tuner_paras": {"names": ["test"],
                                     "initial_values": [10],
                                     "bounds": [(0, 20)]},
                     "goals": {"variable_names": var_names,
                               "statistical_measure": "RMSE",
                               "meas_target_data": {"data": example_hdf,
                                                    "key": "examples"}},
                     "name": "test",
                     "start_time": 0,
                     "stop_time": 10}]
        )[0]

        self.assertIsInstance(cal_class, aixcalibuha.CalibrationClass)
        self.assertIsInstance(cal_class.tuner_paras, aixcalibuha.TunerParas)
        self.assertIsInstance(cal_class.goals, aixcalibuha.Goals)


if __name__ == "__main__":
    unittest.main()
