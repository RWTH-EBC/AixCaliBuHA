"""
Module containing examples on how to use AixCaliBuHA
"""
import pathlib
import sys
from ebcpy import FMU_API
from examples import e2_1_optimization_problem_definition, e2_2_another_data_types_example


def setup_fmu(example="B"):
    """Setup the FMU used in all examples and tests."""
    example_dir = pathlib.Path(__file__).parent

    if example == "A":
        if "win" not in sys.platform:
            raise OSError("Can only run the example type B on windows. "
                          "Select example type A")
        model_name = example_dir.joinpath("model", "HeatPumpSystemWithInput.fmu")
    else:
        if "win" in sys.platform:
            model_name = example_dir.joinpath("model", "PumpAndValve_windows.fmu")
        else:
            model_name = example_dir.joinpath("model", "PumpAndValve_linux.fmu")

    return FMU_API(cd=example_dir.joinpath("testzone"),
                   model_name=model_name,
                   log_fmu=False)


def setup_calibration_classes(example="B", multiple_classes=True):
    """Setup the CalibrationClasses used in all examples and tests."""
    if example == "A":
        return e2_1_optimization_problem_definition.main(multiple_classes=multiple_classes)
    return e2_2_another_data_types_example.setup_calibration_classes()
