"""
Module with functions to convert
certain format into other formats.
"""
import scipy.io as spio
from aixcal import data_types


def convert_hdf_to_mat(filepath, save_path_file, columns=None, key=None, set_time_to_zero=True):
    """
    Function to convert a hdf file to a mat-file readable within Dymola.
    :param filepath: str, os.path.normpath
        String or even os.path.normpath.
        Must point to a valid hdf file.
    :param save_path_file: str, os.path.normpath
        File path and name where to store the output *.mat file.
    :param columns: list,
        A list with names of columns that should be saved to *.mat file.
        If no list is provided, all columns are converted.
    :param key: str,
        The name of the dataframe inside the given hdf-file.
        Only needed if multiple tables are stored within tht given file.
    :param set_time_to_zero: bool (default True),
        If True, the index, which is the time, will start at a zero base.
    :return mat_file:
         Returns the version 4 mat-file

    :Example
    >>> import os
    >>> project_dir = os.path.dirname(os.path.dirname(__file__))
    >>> example_filepath = os.path.normpath(project_dir + "//examples//example_data.hdf")
    >>> save_path = os.path.normpath(project_dir + "//examples//example_data_converted.mat")
    >>> columns = ["sine.y / "]
    >>> convert_hdf_to_mat(example_filepath, save_path, columns=columns, key="trajectories")
    Successfully converted given .hdf-file to .mat-file.
    >>> os.remove(save_path)
    """
    data = data_types.TimeSeriesData(filepath, **{"key": key})
    df = data.df
    if set_time_to_zero:
        df.index = df.index - df.iloc[0].name.to_datetime64()  # Make index zero based
    df['time_vector'] = df.index.total_seconds()  # Copy values of index as seconds into new column
    if not columns:
        columns = df.columns
    # Add column name of time in front of desired columns,
    # since time must be first column in the *.mat file.
    columns = ['time_vector'] + list(columns)
    # Store desired columns in new variable, which is a np.array
    subset = df[columns]
    # Convert np.array into a list and create a dict with 'table' as matrix name
    new_mat = {'table': subset.values.tolist()}
    # Save matrix as a MATLAB *.mat file, which is readable by Modelica.
    spio.savemat(save_path_file, new_mat, format="4")
    print("Successfully converted given .hdf-file to .mat-file.")


if __name__ == '__main__':
    import doctest
    doctest.testmod()
