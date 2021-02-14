"""Test-module for all classes inside
aixcalibuha.sensanalyzer except the modelica based SenAnalyzer, as the setup
is more fitting inside test_modelica_cal_sen"""
import unittest
from aixcalibuha.sensanalyzer import sensitivity_analyzer
from aixcalibuha import TunerParas

class TestSenProblem(unittest.TestCase):
    """Test-class for sensitivity analysis problem class
    """

    def test_sen_problem(self):
        """Test setup of a sensitivity problem"""
        # Test general setup for supported methods
        for method in ["morris", "sobol"]:
            sen_problem = sensitivity_analyzer.SensitivityProblem(method,
                                                                  num_samples=2)
            self.assertIsInstance(sen_problem, sensitivity_analyzer.SensitivityProblem)

        # Test setup with tuner parameters
        tuner_paras = TunerParas(["heatConv_a", "heatConv_b", "C", "m_flow_2"],
                                 [130, 220, 5000, 0.04])
        sen_problem = sensitivity_analyzer.SensitivityProblem("morris", num_samples=2,
                                                              tuner_paras=tuner_paras)
        self.assertIsInstance(sen_problem.problem, dict)

        # Check error if input wrong
        with self.assertRaises(KeyError):
            sensitivity_analyzer.SensitivityProblem("not_supported_method",
                                                    num_samples=2)


if __name__ == "__main__":
    unittest.main()
