import unittest
import os
from aixcal.utils import visualizer
from aixcal.utils import statistics_analyzer


class TestDataTypes(unittest.TestCase):

    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        self.framework_dir = os.path.dirname(os.path.dirname(__file__))
        self.example_dir = os.path.normpath(self.framework_dir + "//examples")

    def test_statistics_analyzer(self):
        """Test class StatisticsAnalyzer"""
        # TODO Implement test
        pass

    def test_visualizer_logger(self):
        """Test class Logger"""
        # TODO Implement test
        pass

    def test_visualizer_visualizer(self):
        """Test class Visualizer"""
        # TODO Implement test
        pass


if __name__ == "__main__":
    unittest.main()
