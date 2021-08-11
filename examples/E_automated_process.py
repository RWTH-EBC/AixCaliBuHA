from examples import setup_fmu, setup_calibration_classes
from examples.C_sensitivity_analysis_example import run_sensitivity_analysis
from examples.D_calibration_example import run_calibration


def main(example="A"):
    # Parameters for sen-analysis:
    sim_api = setup_fmu(example=example)
    calibration_classes, validation_class = setup_calibration_classes(example=example)

    # Sensitivity analysis:
    calibration_classes = run_sensitivity_analysis(sim_api=sim_api,
                                                   cal_classes=calibration_classes)
    # Calibration
    run_calibration(sim_api=sim_api,
                    cal_classes=calibration_classes,
                    validation_class=validation_class)


if __name__ == '__main__':
    main(example="A")
