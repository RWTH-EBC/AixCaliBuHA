"""
Run an example for a calibration using multiprocessing.
Multiprocessing only runs with pymoo as the used framework.
"""


if __name__ == "__main__":
    from examples import setup_fmu, setup_calibration_classes
    from examples.e4_calibration_example import run_calibration
    # Number of logical Processors to run calibration on:
    N_CPU = 5
    # Parameters for sen-analysis:
    EXAMPLE = "A"  # Or choose B
    SIM_API = setup_fmu(example=EXAMPLE, n_cpu=N_CPU)
    CALIBRATION_CLASSES, VALIDATION_CLASS = setup_calibration_classes(
        example=EXAMPLE,
        multiple_classes=False
    )

    # Sensitivity analysis:
    run_calibration(
        sim_api=SIM_API,
        cal_classes=CALIBRATION_CLASSES,
        validation_class=VALIDATION_CLASS,
        framework="pymoo",
        method="GA"
    )
