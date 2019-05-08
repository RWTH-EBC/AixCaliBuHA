import unittest
import os
from aixcal.simulationapi import dymola_api
from aixcal.simulationapi import py_fmi


class TestDataTypes(unittest.TestCase):

    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        self.framework_dir = os.path.dirname(os.path.dirname(__file__))
        self.example_dir = os.path.normpath(self.framework_dir + "//examples")


if __name__ == "__main__":
    unittest.main()
