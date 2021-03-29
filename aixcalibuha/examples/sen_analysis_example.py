"""
Example file for the senanalyzer package. The usage of modules and classes inside
the senanalyzer package should be clear when looking at the examples.
If not, please raise an issue.
"""

from ebcpy.examples import dymola_api_example
from aixcalibuha.sensanalyzer import MorrisAnalyzer
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

    sen_analyzer = MorrisAnalyzer(
        sim_api=sim_api,
        statistical_measure=stat_measure,
        num_samples=1,
        cd=sim_api.cd,
        analysis_variable='mu_star'
    )

    print('Unsorted classes order: ')
    print(', '.join([c.name for c in cal_classes]))
    sorted_classes = sen_analyzer.automatic_run(calibration_classes=cal_classes)
    print('Sorted classes after SA: ')
    print(', '.join([c.name for c in sorted_classes]))

    return sorted_classes


if __name__ == "__main__":
    # Parameters for sen-analysis:
    STATISTICAL_MEASURE = "RMSE"

    DYM_API = dymola_api_example.setup_dymola_api()
    CALIBRATION_CLASSES = cal_classes_example.setup_calibration_classes()

    # Sensitivity analysis:
    CALIBRATION_CLASSES = example_sensitivity_analysis(DYM_API,
                                                       CALIBRATION_CLASSES,
                                                       STATISTICAL_MEASURE)
