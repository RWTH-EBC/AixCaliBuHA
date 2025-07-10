"""
Package containing utility functions used in different packages.
Contains a statistics analyzer and a visualizer.
"""
import os
from typing import Union, List
from pathlib import Path
from aixcalibuha import CalibrationClass
from ebcpy import TimeSeriesData
import pandas as pd


def convert_mat_to_suffix(mat_result_file, variable_names, suffix_files, parquet_engine='pyarrow', compression='snappy'):
    """
    Postprocess the mat result files.

    :param str mat_result_file: The path to the MATLAB result file.
    :param List[str] variable_names: The names of the variables to extract.
    :param str suffix_files: The suffix for the output files.
    :param str parquet_engine: The engine to use for saving parquet files.
    """
    df = TimeSeriesData(mat_result_file, variable_names=variable_names).to_df()
    df_path = Path(mat_result_file).with_suffix("." + suffix_files)
    if suffix_files == "csv":
        df.to_csv(df_path)
    elif suffix_files == "parquet":
        df_for_disk = df.copy()
        for col in df_for_disk.columns:
            if isinstance(df_for_disk[col].dtype, pd.SparseDtype):
                df_for_disk[col] = df_for_disk[col].sparse.to_dense()
        df.to_parquet(
            df_path,
            engine=parquet_engine,
            compression=compression,
            index=True
        )
    else:
        raise ValueError(f"Unsupported file suffix: {suffix_files}. "
                         "Supported suffixes are 'csv' and 'parquet'.")
    os.remove(mat_result_file)
    return df_path


def empty_postprocessing(mat_result, **_kwargs):
    return mat_result


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
    
class MaxTimeReached(Exception):
    """
    Exception raised for when the calibration
    ends because the maximum calibration time is reached.
    """
