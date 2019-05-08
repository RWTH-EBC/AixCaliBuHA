import unittest, os
from aixcal import data_types
import pandas as pd


class TestDataTypes(unittest.TestCase):

    def setUp(self):
        """Called before every test.
        Define example paths and parameters used in all test-functions.
        """
        self.framework_dir = os.path.dirname(os.path.dirname(__file__))
        self.example_dir = os.path.normpath(self.framework_dir + "//examples")
        self.example_data_hdf_path = os.path.normpath(self.example_dir + "//example_data.hdf")
        self.example_data_csv_path = os.path.normpath(self.example_dir + "//example_data.CSV")
        self.example_data_mat_path = os.path.normpath(self.example_dir + "//example_data.mat")

    def test_time_series_data(self):
        """Test the class TimeSeriesData"""
        # Test if wrong input leads to FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            data_types.TimeSeriesData("Z:\\this_will_never_be_a_file_path.hdf")
        # Test if wrong file-ending leads to TypeError
        with self.assertRaises(TypeError):
            a_python_file = __file__
            data_types.TimeSeriesData(a_python_file)
        # If no key is provided, a KeyError has to be raised
        with self.assertRaises(KeyError):
            data_types.TimeSeriesData(self.example_data_hdf_path)
        with self.assertRaises(KeyError):
            data_types.TimeSeriesData(self.example_data_hdf_path, key="wrong_key")
        # Correctly load the .hdf:
        time_series_data = data_types.TimeSeriesData(self.example_data_hdf_path, key="parameters")
        self.assertIsInstance(
            time_series_data.df,
            type(pd.DataFrame()))
        # Correctly load the .csv:
        time_series_data = data_types.TimeSeriesData(self.example_data_csv_path, sep=",")
        self.assertIsInstance(
            time_series_data.df,
            type(pd.DataFrame()))
        # Correctly load the .mat:
        time_series_data = data_types.TimeSeriesData(self.example_data_mat_path)
        self.assertIsInstance(
            time_series_data.df,
            type(pd.DataFrame()))

    def test_meas_target_data(self):
        """Test the class MeasTargetData.
        For a detailed test of this class, see base-class test_time_series_data()"""
        meas_target_data = data_types.MeasTargetData(self.example_data_hdf_path, key="trajectories")
        self.assertEqual(meas_target_data.data_type, "MeasTargetData")

    def test_meas_input_data(self):
        """Test the class MeasInputData.
        For a detailed test of this class, see base-class test_time_series_data()"""
        meas_input_data = data_types.MeasInputData(self.example_data_hdf_path, key="trajectories")
        self.assertEqual(meas_input_data.data_type, "MeasInputData")

    def test_sim_target_data(self):
        """Test the class SimTargetData.
        For a detailed test of this class, see base-class test_time_series_data()"""
        sim_target_data = data_types.SimTargetData(self.example_data_hdf_path, key="trajectories")
        self.assertEqual(sim_target_data.data_type, "SimTargetData")

    def test_tuner_para(self):
        """Test the class TunerPara"""
        # TODO Once a decision is made on the structure of tunerPara-Class, implement the test.
        pass

    def test_goal(self):
        """Test the class Goal"""
        # TODO Once a decision is made on the structure of goal-Class, implement the test.
        pass

    def test_calibration_class(self):
        """Test the class CalibrationClass"""
        with self.assertRaises(ValueError):
            # Test if start-time higher than stop-time raises an error.
            data_types.CalibrationClass("dummy", 100, 50)
        with self.assertRaises(TypeError):
            # Test if a given name not equal to string raises an error.
            not_a_string = 1
            data_types.CalibrationClass(not_a_string, 0, 10)

        # Test set_functions for goals and tuner parameters
        dummy_tuner_para = "not TunerPara-Class"
        dummy_goal = "not Goal-Class"
        dummy_cal_class = data_types.CalibrationClass("dummy", 0, 10)
        with self.assertRaises(TypeError):
            dummy_cal_class.set_tuner_para(dummy_tuner_para)
        with self.assertRaises(TypeError):
            dummy_cal_class.set_goals(dummy_goal)

    def test_get_keys_of_hdf_file(self):
        """Test the function get_keys_of_hdf_file.
        Check the keys of the file with e.g. the SDFEditor and
        use those keys as a reference list.
        """
        reference_list = ['parameters', 'trajectories']
        return_val = data_types.get_keys_of_hdf_file(self.example_data_hdf_path)
        self.assertListEqual(return_val, reference_list)


if __name__ == "__main__":
    unittest.main()
