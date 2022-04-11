# # Example 5 Automated process

# Goals of this part of the examples:
# 1. Learn how to run everything in one script
#
# Start by importing everything
from examples.e3_sensitivity_analysis_example import run_sensitivity_analysis
from examples.e4_calibration_example import run_calibration


def main(
        examples_dir,
        example: str = "A",
        n_cpu: int = 1
):
    """
    Arguments of this example:

    :param [pathlib.Path, str] examples_dir:
        Path to the examples folder of AixCaliBuHA
    :param str example:
        Whether to use example A (requires windows) or B.
        Default is "A"
    :param int n_cpu:
        Number of cores to use

    """
    # First we run the sensitivity analysis:
    calibration_classes, sim_api = run_sensitivity_analysis(
        examples_dir=examples_dir, example=example, n_cpu=n_cpu
    )
    # Then the calibration and validation
    run_calibration(
        examples_dir=examples_dir,
        example=example,
        sim_api=sim_api,
        cal_classes=calibration_classes
    )


if __name__ == '__main__':
    import pathlib
    EXAMPLE = "B"
    main(examples_dir=pathlib.Path(__file__).parent, example=EXAMPLE)
