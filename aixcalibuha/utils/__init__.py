"""
Package containing utility functions used in different packages.
Contains a statistics analyzer and a visualizer.
"""
from typing import Union, List
from aixcalibuha import CalibrationClass


def validate_cal_class_input(
        calibration_classes: Union[CalibrationClass, List[CalibrationClass]]
) -> List[CalibrationClass]:
    """Check if given list contains only CalibrationClass objects or is one
    and return a list in both cases. Else raise an error"""
    if isinstance(calibration_classes, list):
        for cal_class in calibration_classes:
            if not isinstance(cal_class, CalibrationClass):
                raise TypeError(f"calibration_classes is of type {type(cal_class).__name__} "
                                f"but should be CalibrationClass")
    elif isinstance(calibration_classes, CalibrationClass):
        calibration_classes = [calibration_classes]
    else:
        raise TypeError(f"calibration_classes is of type {type(calibration_classes).__name__} "
                        f"but should be CalibrationClass or list")
    return calibration_classes


class MaxIterationsReached(Exception):
    """
    Exception raised for when the calibration
    ends because the maximum number of
    allowed iterations is reached.
    """
