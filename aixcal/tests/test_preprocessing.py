"""Test-module for all classes inside
aixcal.preprocessing."""
import unittest
import os
from datetime import datetime
import scipy.io as spio
import numpy as np
import pandas as pd
from aixcal.preprocessing import conversion
from aixcal.preprocessing import preprocessing


class TestConversion(unittest.TestCase):
    """Test-class for preprocessing."""

    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        self.framework_dir = os.path.dirname(os.path.dirname(__file__))
        self.example_dir = os.path.normpath(self.framework_dir + "//examples")
        self.example_data_hdf_path = os.path.normpath(self.example_dir + "//example_data.hdf")

    def test_conversion_hdf_to_mat(self):
        """Test function conversion.convert_hdf_to_mat().
        For an example, see the doctest in the function."""
        # First convert the file
        save_path = os.path.normpath(self.example_dir + "//example_data_converted.mat")
        columns = ["sine.y / "]
        # Test both conversion with specification of columns and without passing the names.
        for col in [columns, None]:
            res, filepath_mat = conversion.convert_hdf_to_mat(self.example_data_hdf_path,
                                                              save_path,
                                                              columns=col,
                                                              key="trajectories")
            # Check if successfully converted
            self.assertTrue(res)
            # Check if converted file exists
            self.assertTrue(os.path.isfile(filepath_mat))
            # Check if converted filepath is provided filepath
            self.assertEqual(filepath_mat, save_path)
            # Now check if the created mat-file can be used.
            self.assertIsInstance(spio.loadmat(save_path), dict)
            # Remove converted file again
            os.remove(save_path)


class TestPreProcessing(unittest.TestCase):
    """Test-class for preprocessing."""

    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        self.framework_dir = os.path.dirname(os.path.dirname(__file__))
        self.example_dir = os.path.normpath(self.framework_dir + "//examples")
        self.example_data_hdf_path = os.path.normpath(self.example_dir + "//example_data.hdf")

    def test_build_average_on_duplicate_rows(self):
        """Test function of preprocessing.build_average_on_duplicate_rows().
        For an example, see the doctest in the function."""
        # Choose random number to check if function works in every dimension
        dim = np.random.randint(1, 1000)
        vals = np.random.rand(dim)
        # instantiate df with index all 1
        df = pd.DataFrame({"idx": np.ones(dim), "val": vals}).set_index("idx")
        df = preprocessing.build_average_on_duplicate_rows(df)
        # Check if the length has been reduced to 1
        self.assertEqual(len(df), 1)
        # Check if the average is computed correctly
        self.assertEqual(df.iloc[0].val, np.average(vals))

    def test_convert_index_to_datetime_index(self):
        """Test function of preprocessing.convert_index_to_datetime_index().
        For an example, see the doctest in the function."""

        dim = np.random.randint(1, 10000)
        df = pd.DataFrame(np.random.rand(dim, 4), columns=list('ABCD'))
        df_temp = preprocessing.convert_index_to_datetime_index(df)
        # Check if index is correctly created
        self.assertIsInstance(df_temp.index, pd.DatetimeIndex)
        # Check different unit-formats:
        for unit in ["ms", "s", "h", "d", "min"]:
            df_temp = preprocessing.convert_index_to_datetime_index(df,
                                                                    unit_of_index=unit)
        # Test different datetime:
        df_temp = preprocessing.convert_index_to_datetime_index(df, origin=datetime(2007, 1, 1))

    def test_clean_and_space_equally_time_series(self):
        """Test function of preprocessing.clean_and_space_equally_time_series().
        For an example, see the doctest in the function."""
        pass

    def test_low_pass_filter(self):
        """Test function of preprocessing.low_pass_filter().
        For an example, see the doctest in the function."""
        pass

    def test_moving_average(self):
        """Test function of preprocessing.moving_average().
        For an example, see the doctest in the function."""
        pass

    def test_create_on_off_signal(self):
        """Test function of preprocessing.create_on_off_signal().
        For an example, see the doctest in the function."""
        df = pd.DataFrame()
        with self.assertRaises(IndexError):
            # Give too many new names
            preprocessing.create_on_off_signal(df, col_names=["Dummy"], threshold=None,
                                               col_names_new=["One", "too much"])
        with self.assertRaises(IndexError):
            # Too many thresholds given
            preprocessing.create_on_off_signal(df, col_names=["Dummy"],
                                               threshold=[1, 2, 3, 4],
                                               col_names_new=["Dummy_signal"])
        time_df = pd.DataFrame({"dummy_P_el": np.sin(np.linspace(-20, 20, 100))*100})
        df = preprocessing.create_on_off_signal(time_df,
                                                col_names=["dummy_P_el"],
                                                threshold=25,
                                                col_names_new=["dummy_signal"])
        self.assertIsInstance(df["dummy_signal"], pd.Series)
        self.assertIsInstance(df, pd.DataFrame)

    def test_number_lines_totally_na(self):
        """Test function of preprocessing.number_lines_totally_na().
        For an example, see the doctest in the function."""
        dim = np.random.randint(100)
        nan_col = [np.NaN for i in range(dim)]
        col = [i for i in range(dim)]
        df_nan = pd.DataFrame({"col_1": nan_col, "col_2": nan_col})
        df_normal = pd.DataFrame({"col_1": nan_col, "col_2": col})
        self.assertEqual(preprocessing.number_lines_totally_na(df_nan), dim)
        self.assertEqual(preprocessing.number_lines_totally_na(df_normal), 0)

    def test_z_score(self):
        """Test function of preprocessing.z_score().
        For an example, see the doctest in the function."""
        pass

    def test_modified_z_score(self):
        """Test function of preprocessing.modified_z_score().
        For an example, see the doctest in the function."""
        pass

    def test_interquartile_range(self):
        """Test function of preprocessing.interquartile_range().
        For an example, see the doctest in the function."""
        pass

    def test_cross_validation(self):
        """Test function of preprocessing.cross_validation().
        For an example, see the doctest in the function."""
        pass


if __name__ == "__main__":
    unittest.main()
