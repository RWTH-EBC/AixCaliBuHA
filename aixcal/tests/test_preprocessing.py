"""Test-module for all classes inside
aixcal.preprocessing."""
import unittest
import os
import scipy.io as spio
from aixcal.preprocessing import conversion


class TestPreProcessing(unittest.TestCase):
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
        res, filepath_mat = conversion.convert_hdf_to_mat(self.example_data_hdf_path,
                                      save_path,
                                      columns=columns,
                                      key="trajectories")
        self.assertTrue(res)  # Check if successfully converted
        self.assertTrue(os.path.isfile(filepath_mat))  # Check if converted file exists
        self.assertEqual(filepath_mat, save_path)  # Check if converted filepath is provided filepath
        # Now check if the created mat-file can be used.
        self.assertIsInstance(spio.loadmat(save_path), dict)
        # Remove converted file again
        os.remove(save_path)


if __name__ == "__main__":
    unittest.main()
