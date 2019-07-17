import os
from aixcal import data_types
from aixcal.sensanalyzer import sensitivity_analyzer
import pandas as pd
from aixcal.examples import data_types_example, dymola_api_example


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

    :param aixcal.simulationapi.SimulationAPI sim_api:
        Simulation api to run the simulation for the sensitivtiy analysis
    :param list cal_classes:
        List of :meth:`calibration-class<aixcal.data_types.CalibrationClass>`
        objects to be analyzed.
    :param str stat_measure:
        The statistical measure, one of the possible of
        the :meth:`statistics analyzer<aixcal.utils.statistics_analyzer.StatisticsAnalyzer>`
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

    cal_classes = sen_analyzer.automatic_select(cal_classes,
                                                sen_result,
                                                threshold=1)

    return cal_classes


if __name__ == "__main__":
    # Define root of this example-file
    filepath = os.path.dirname(__file__)

    # %% Define path in which you want ot work:
    cd = os.getcwd()

    # Load measured data files
    measTargetData = data_types.MeasTargetData(filepath + "//data//measTargetData.mat")

    # Parameters for sen-analysis:
    statistical_measure = "RMSE"

    dym_api = dymola_api_example.setup_dymola_api()
    calibration_classes = data_types_example.setup_calibration_classes()

    # %% Sensitivity analysis:
    calibration_classes = example_sensitivity_analysis(dym_api,
                                                       calibration_classes,
                                                       statistical_measure)
