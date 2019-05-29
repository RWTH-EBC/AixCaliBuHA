"""Test-module for all classes inside
aixcal.classifier."""
import unittest
import os


class TestClassifier(unittest.TestCase):
    """
    Test the classifier class of aixcal
    """

    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        self.framework_dir = os.path.dirname(os.path.dirname(__file__))
        self.example_dir = os.path.normpath(self.framework_dir + "//examples")


if __name__ == "__main__":
    unittest.main()
