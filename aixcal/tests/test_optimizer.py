"""Test-module for all classes inside
aixcal.optimizer."""

import unittest
import os
import shutil
from aixcal.optimizer import Optimizer, Calibrator
import numpy as np


class TestOptimizer(unittest.TestCase):
    """Test-class for optimizer class"""

    def setUp(self):
        """Called before every test.
        Used to setup relevant paths and APIs etc."""
        self.framework_dir = os.path.dirname(os.path.dirname(__file__))
        self.example_dir = os.path.normpath(self.framework_dir + "//examples")
        self.example_opt_dir = os.path.normpath(self.example_dir + "//test_optimization")
        self.supported_frameworks = ["scipy", "dlib"]

    def test_optimizer_base_class(self):
        """Test-case for the base-class for optimization."""
        opt = Optimizer(self.example_opt_dir)
        for _framework in self.supported_frameworks:
            if _framework == "scipy":
                reference_function = opt._minimize_scipy
            elif _framework == "dlib":
                reference_function = opt._minimize_dlib
            opt.run("dummy_method", _framework)
            self.assertEqual(opt._minimize_func, reference_function)
        with self.assertRaises(TypeError):
            opt.run("dummy_method", "not_supported_framework")

    def test_custom_optimizer(self):
        """Test-case for the customization of the optimizer-base-class."""
        # Define the customized class:
        class CustomOptimizer(Optimizer):

            x_goal = np.random.rand(3)

            def obj(self, xk, *args):
                x = np.linspace(-2, 2, 100)
                a, b, c = xk[0], xk[1], xk[2]
                _a, _b, _c = self.x_goal[0], self.x_goal[1], self.x_goal[2]
                quadratic_func_should = _a*x**2 + _b*x + _c
                quadratic_func_is = a*x**2 + b*x + c
                # Return the MAE of the quadratic function.
                return np.sum(np.abs(quadratic_func_should - quadratic_func_is))

            def run(self, method, framework):
                super().run(method, framework)
                self.x0 = np.array([0, 0, 0])
                return self._minimize_func(method)

        my_custom_optimizer = CustomOptimizer(self.example_opt_dir)
        res = my_custom_optimizer.run("L-BFGS-B", "scipy")
        delta_solution = np.sum(res.x - my_custom_optimizer.x_goal)
        self.assertEqual(0.0, np.round(delta_solution, 3))

    def tearDown(self):
        """Remove all created folders while calibrating."""
        shutil.rmtree(self.example_opt_dir)


if __name__ == "__main__":
    unittest.main()
