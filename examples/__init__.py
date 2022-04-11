"""
Module containing examples on how to use AixCaliBuHA
"""
import pathlib
import sys
from ebcpy import FMU_API
from examples import e2_A_optimization_problem_definition, e2_B_optimization_problem_definition


def setup_fmu(examples_dir, example="B", n_cpu=1):
    """
    Setup the FMU used in all examples and tests.

    :param str examples_dir:
        Path to the examples folder of AixCaliBuHA
    :param str example:
        Which example to run, "A" or "B"
    :param int n_cpu:
        Number of cores to use
    """
    examples_dir = pathlib.Path(examples_dir)
    if example == "A":
        if "win" not in sys.platform:
            raise OSError("Can only run the example type B on windows. "
                          "Select example type A")
        model_name = examples_dir.joinpath("model", "HeatPumpSystemWithInput.fmu")
    elif example == "B":
        if "win" in sys.platform:
            model_name = examples_dir.joinpath("model", "PumpAndValve_windows.fmu")
        else:
            model_name = examples_dir.joinpath("model", "PumpAndValve_linux.fmu")
    else:
        raise ValueError("Only example 'A' and 'B' are available")

    return FMU_API(cd=examples_dir.joinpath("testzone"),
                   model_name=model_name,
                   log_fmu=False,
                   n_cpu=n_cpu)


def setup_calibration_classes(examples_dir, example="B", multiple_classes=True):
    """Setup the CalibrationClasses used in all examples and tests."""
    if example == "A":
        return e2_A_optimization_problem_definition.main(
            examples_dir=examples_dir, multiple_classes=multiple_classes
        )
    if example == "B":
        return e2_B_optimization_problem_definition.main(
            examples_dir=examples_dir, multiple_classes=multiple_classes
        )
    raise ValueError("Only example 'A' and 'B' are available")
