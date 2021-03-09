"""
Example file for the senanalyzer package. The usage of modules and classes inside
the senanalyzer package should be clear when looking at the examples.
If not, please raise an issue.
"""

from ebcpy.examples import dymola_api_example
import pandas as pd
from aixcalibuha.sensanalyzer import sensitivity_analyzer
from aixcalibuha.examples import cal_classes_example


def example_sensitivity_analysis(sim_api, cal_classes, stat_measure):
    """
    Example process of a sensitivity analysis.
    First, the sensitivity problem is constructed, in this example
    the `morris` method is chosen.
    Afterwards, the sen_analyzer class is instantiated to run the
    sensitivity analysis in the next step.
    The result of this analysis is then printed to the user.
    The automatic_select function is presented as-well, using a threshold of 1
    and the default `mu_star` criterion.

    :param aixcalibuha.simulationapi.SimulationAPI sim_api:
        Simulation api to run the simulation for the sensitivtiy analysis
    :param list cal_classes:
        List of :meth:`calibration-class<aixcalibuha.data_types.CalibrationClass>`
        objects to be analyzed.
    :param str stat_measure:
        The statistical measure, one of the possible of
        the :meth:`statistics analyzer<ebcpy.utils.statistics_analyzer.StatisticsAnalyzer>`
    :return: A list calibration classes
    :rtype: list
    """
    # Setup the class
    sen_problem = sensitivity_analyzer.SensitivityProblem("morris",
                                                          num_samples=2)

    sen_analyzer = sensitivity_analyzer.SenAnalyzer(sim_api.cd,
                                                    simulation_api=sim_api,
                                                    sensitivity_problem=sen_problem,
                                                    calibration_classes=cal_classes,
                                                    statistical_measure=stat_measure)

    # Choose initial_values and set boundaries to tuner_parameters
    # Evaluate which tuner_para has influence on what class
    sen_result = sen_analyzer.run()

    for result in sen_result:
        print(pd.DataFrame(result))

    cal_classes = sen_analyzer.select_by_threshold(sen_analyzer.calibration_classes,
                                                   sen_result,
                                                   threshold=1)

    return cal_classes


if __name__ == "__main__":
    # Parameters for sen-analysis:
    STATISTICAL_MEASURE = "RMSE"

    DYM_API = dymola_api_example.setup_dymola_api()
    CALIBRATION_CLASSES = cal_classes_example.setup_calibration_classes()

    # %% Sensitivity analysis:
    CALIBRATION_CLASSES = example_sensitivity_analysis(DYM_API,
                                                       CALIBRATION_CLASSES,
                                                       STATISTICAL_MEASURE)
