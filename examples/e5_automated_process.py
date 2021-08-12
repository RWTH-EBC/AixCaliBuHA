"""
Goals of this part of the examples:

1. Learn how to run everything in one script
"""

from examples import setup_fmu, setup_calibration_classes
from examples.e3_sensitivity_analysis_example import run_sensitivity_analysis
from examples.e4_calibration_example import run_calibration


def main(example="A"):
    # Parameters for sen-analysis:
    sim_api = setup_fmu(example=example)
    calibration_classes, validation_class = setup_calibration_classes(
        example=example,
        multiple_classes=True
    )

    # Sensitivity analysis:
    calibration_classes = run_sensitivity_analysis(sim_api=sim_api,
                                                   cal_classes=calibration_classes)
    # Calibration
    run_calibration(sim_api=sim_api,
                    cal_classes=calibration_classes,
                    validation_class=validation_class)


if __name__ == '__main__':
    main(example="A")
