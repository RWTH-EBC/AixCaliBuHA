"""
Run an example for a calibration using multiprocessing.
Multiprocessing only runs with pymoo as the used framework.
"""


if __name__ == "__main__":
    import pathlib
    from examples.e4_calibration_example import run_calibration
    # Number of logical Processors to run calibration on:
    N_CPU = 5
    # Parameters for sen-analysis:
    EXAMPLE = "A"  # Or choose B

    # Sensitivity analysis:
    run_calibration(
        examples_dir=pathlib.Path(__file__).parent,
        example=EXAMPLE,
        n_cpu=N_CPU,
        framework="pymoo",
        method="GA"
    )
