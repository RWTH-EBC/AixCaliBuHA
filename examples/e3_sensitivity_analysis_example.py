# # Example 3 sensitivity analysis

# Goals of this part of the examples:
# 1. Learn how to execute a sensitivity analysis
# 2. Learn how to automatically select sensitive tuner parameters
#
# Import a valid analyzer, e.g. `SobolAnalyzer`
from aixcalibuha import SobolAnalyzer


def run_sensitivity_analysis(example="B"):
    """
    Example process of a sensitivity analysis.
    First, the sensitivity problem is constructed, in this example
    the `morris` method is chosen.
    Afterwards, the sen_analyzer class is instantiated to run the
    sensitivity analysis in the next step.
    The result of this analysis is then printed to the user.
    The automatic_select function is presented as-well, using a threshold of 1
    and the default `mu_star` criterion.

    :param str example:
        Which example to run, "A" or "B"

    :return: A list of calibration classes
    :rtype: list
    """
    # ## Setup
    # Setup the class according to the documentation.
    # You just have to pass a valid simulation api and
    # some further settings for the analysis.
    # Let's thus first load the necessary simulation api:
    from examples import setup_fmu, setup_calibration_classes
    sim_api = setup_fmu(example=example)

    sen_analyzer = SobolAnalyzer(
            sim_api=sim_api,
            num_samples=10,
            cd=sim_api.cd,
            analysis_variable='S1'
        )
    # Now perform the analysis for the one of the given calibration classes.
    calibration_classes = setup_calibration_classes(example=example)[0]

    result, classes = sen_analyzer.run(calibration_classes=calibration_classes)
    print("Result of the sensitivity analysis")
    print(result)
    # For each given class, you should see the given tuner parameters
    # and the sensitivity according to the selected method from the SALib.
    # Let's remove some less sensitive parameters based on some threshold
    # to remove complexity from our calibration problem:
    print("Selecting relevant tuner-parameters using a fixed threshold:")
    sen_analyzer.select_by_threshold(calibration_classes=classes,
                                     result=result,
                                     threshold=0.01)
    for cal_class in classes:
        print(f"Class '{cal_class.name}' with parameters:\n{cal_class.tuner_paras}")
    # Return the classes and the sim_api to later perform an automated process in example 5
    return classes, sim_api


if __name__ == "__main__":
    # Parameters for sen-analysis:
    EXAMPLE = "B"  # Or choose A
    # Sensitivity analysis:
    run_sensitivity_analysis(example=EXAMPLE)
