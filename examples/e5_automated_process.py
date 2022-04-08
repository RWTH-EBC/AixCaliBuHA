# # Example 5 Automated process

# Goals of this part of the examples:
# 1. Learn how to run everything in one script
#
# Start by importing everything
from examples import setup_fmu, setup_calibration_classes
from examples.e3_sensitivity_analysis_example import run_sensitivity_analysis
from examples.e4_calibration_example import run_calibration


def main(example="A"):
    """
    Arguments of this example:

    :param str example:
        Whether to use example A (requires windows) or B.
        Default is "A"
    """
    # First we run the sensitivity analysis:
    calibration_classes, sim_api = run_sensitivity_analysis(example=example)
    # Then the calibration and validation
    run_calibration(example=example,
                    sim_api=sim_api,
                    cal_classes=calibration_classes)


if __name__ == '__main__':
    main(example="B")
