"""
Example file for the senstivity_analyzer package. The usage of modules and classes inside
the senanalyzer package should be clear when looking at the examples.
If not, please raise an issue.

Goals of this part of the examples:
1. Learn how to execute a sensitivity analysis
2. Learn how to automatically select sensitive tuner parameters
"""
from aixcalibuha import SobolAnalyzer


def run_sensitivity_analysis(sim_api, cal_classes):
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
    :return: A list calibration classes
    :rtype: list
    """
    # Setup the class

    sen_analyzer = SobolAnalyzer(
            sim_api=sim_api,
            num_samples=10,
            cd=sim_api.cd,
            analysis_variable='S1'
        )

    result, classes = sen_analyzer.run(calibration_classes=cal_classes)
    print("Result of the sensitivity analysis")
    print(result)
    print("Selecting relevant tuner-parameters using a fixed threshold:")
    sen_analyzer.select_by_threshold(calibration_classes=cal_classes,
                                     result=result,
                                     threshold=0.01)
    for cal_class in cal_classes:
        print(f"Class '{cal_class.name}' with parameters:\n{cal_class.tuner_paras}")
    return classes


if __name__ == "__main__":
    from examples import setup_fmu, setup_calibration_classes
    # Parameters for sen-analysis:
    EXAMPLE = "B"  # Or choose A
    N_CPU = 2
    SIM_API = setup_fmu(example=EXAMPLE, n_cpu=N_CPU)
    CALIBRATION_CLASSES = setup_calibration_classes(example=EXAMPLE)[0]

    # Sensitivity analysis:
    CALIBRATION_CLASSES = run_sensitivity_analysis(sim_api=SIM_API,
                                                   cal_classes=CALIBRATION_CLASSES)
