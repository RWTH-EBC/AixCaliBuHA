"""Test-module for all classes inside
aixcal.sensanalyzer."""
import unittest
import os


class TestSenAnalyzer(unittest.TestCase):
    """Test-class for sensitivity analysis
    """
    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        self.framework_dir = os.path.dirname(os.path.dirname(__file__))
        self.example_dir = os.path.normpath(self.framework_dir + "//examples")


if __name__ == "__main__":
    unittest.main()
