"""
Module containing examples on how to use AixCaliBuHA
"""
import pathlib
import sys
from ebcpy import FMU_API


def setup_fmu():
    """Setup the FMU used in all examples and tests."""
    example_dir = pathlib.Path(__file__).parent

    if "win" in sys.platform:
        model_name = example_dir.joinpath("model", "PumpAndValve_windows.fmu")
    else:
        model_name = example_dir.joinpath("model", "PumpAndValve_linux.fmu")

    return FMU_API(cd=example_dir.joinpath("testzone"),
                   model_name=model_name)
