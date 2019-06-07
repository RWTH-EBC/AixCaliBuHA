"""Module with static functions used to preprocess or alter
data, maily in the format of datafames or np.arrays."""
from datetime import datetime
from scipy import signal
import numpy as np
import pandas as pd


def build_average_on_duplicate_rows(df):
    """
    If the dataframe has duplicate-indexes, the average
    value of all those indexes is calculated and given to
    the first occurrence of this duplicate index. Therefore,
    any dataFrame should be already sorted before calling this
    function.
    Example:
    Dataframe e.g.:
    2000-01-01 00:00:02.340   6.500    ...            1.0
    2000-01-01 00:00:02.340   7.000    ...            1.0
    will become:
    2000-01-01 00:00:02.340   6.750    ...            1.0

    :param df: pd.DataFame
        DataFrame with the data to process
    :return: pd.DataFame
    """
    # Find entries that are exactly the same timestamp
    double_ind = df.index[df.index.duplicated()].unique()
    # Calculate the mean value
    mean_values = []
    for item in double_ind:
        mean_values.append(df.loc[item].values.mean(axis=0))
    # Delete duplicate indices
    df_dropped = df[~df.index.duplicated(keep='first')].copy()

    # Set mean values in rows that were duplicates before
    for idx, values in zip(double_ind, mean_values):
        df_dropped.loc[idx] = values

    return df_dropped


def convert_index_to_datetime_index(df, unit_of_index="s"):
    """
    Converts the index of the given DataFrame to a
    pandas.core.indexes.datetimes.DatetimeIndex.
    :param df: pd.DataFrame
        dataframe with index not being a DateTime.
        Only numeric indexes are supported. Every integer
        is interpreted with the given unit, standard form
        is in seocnds.
    :param unit_of_index: str, default 's'
        The unit of the given index. Used to convert to
        total_seconds later on.
    :return: df
        DataFrame with correct index for usage in this
        framework.
    """
    # Check for unit of given index. Maybe one uses hour-based data.
    _unit_conversion_to_seconds = {"ms": 1e-3,
                                   "s": 1,
                                   "min": 1/60,
                                   "h": 1/3600,
                                   "d": 1/86400}
    if unit_of_index not in _unit_conversion_to_seconds:
        raise ValueError("Given unit_of_index is not supported.")
    _unit_factor_to_seconds = _unit_conversion_to_seconds.get(unit_of_index)

    #Convert
    old_index = df.index.copy()
    # Check if already converted:
    if isinstance(old_index, pd.DatetimeIndex):
        return df
    # Convert strings to numeric values.
    old_index = pd.to_numeric(old_index)
    # Convert to seconds.
    old_index *= _unit_factor_to_seconds
    # Alter the index
    df.index = pd.to_datetime(old_index, unit="s", origin=datetime.now())

    return df


def clean_and_space_equally_time_series(df, desired_freq):
    """
    Function for cleaning of the given dataFrame and interpolating
    based on the the given desired frequency. Linear interpolation
    is used.
    :param df: pd.DataFrame
        Unclean DataFrame. Needs to have a pd.DateTimeIndex
    :param desired_freq: float
        Frequency to determine number of elements in processed dataframe
    :return: pd.DataFrame
        Cleaned and equally spaced data-frame
    """
    # Convert indexes to datetime_index:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame needs a DateTimeIndex for executing this function."
                        "Call convert_index_to_datetime_index() to convert any index to "
                        "a DateTimeIndex")
    #%% Check DataFrame for NANs
    # Create a pandas Series with number of invalid values for each column of df
    series_with_na = df.isnull().sum()
    for name in series_with_na.index:
        if series_with_na.loc[name] > 0:
            # Print only columns with invalid values
            print("{} has following number of invalid "
                  "values\n {}".format(name, str(series_with_na.loc[name])))
    # Drop all rows where at least one NA exists
    df.dropna(how='any', inplace=True)

    # Check if DataFrame still has non-numeric-values:
    if not all(df.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all())):
        raise ValueError("Given DataFrame contains non-numeric values.")

    # Merge duplicate rows using mean.
    df = build_average_on_duplicate_rows(df)

    #%% Re-sampling to new frequency with linear interpolation
    # Create new equally spaced DatetimeIndex. Last entry is always < df.index[-1]
    time_index = pd.date_range(start=df.index[0], end=df.index[-1], freq=desired_freq)
    # Create an empty data frame
    df_time_temp = pd.DataFrame(index=time_index)

    # Insert temporary time_index into df. fill_value = 0 can only be used,
    # since all NaNs should be eliminated prior
    df = df.radd(df_time_temp, axis='index', fill_value=0)
    del df_time_temp

    # Interpolate linearly according to time index
    df.interpolate(method='time', axis=0, inplace=True)
    # Determine Timedelta between current first index entry
    # in df and the first index entry that would be created
    # when applying df.resample() without loffset
    delta_time = df.index[0] - df.resample(rule=desired_freq).first().first(desired_freq).index[0]
    # Resample to equally spaced index.
    # All fields should already have a value. Thus NaNs and maybe +/- infs
    # should have been filtered beforehand.
    df = df.resample(rule=desired_freq, loffset=delta_time).first()
    del delta_time

    return df


def low_pass_filter(data, crit_freq, filter_order):
    """
    Create a low pass filter with given order and frequency.
    :param data: numpy.ndarray,
        For dataframe e.g. df['a_col_name'].values
    :param crit_freq: float,
    :param filter_order: int,
    :return: numpy.ndarray,
    """
    _filter_order = int(filter_order)
    numerator, denominator = signal.butter(N=_filter_order, Wn=crit_freq,
                                           btype='low', analog=False, output='ba')
    output = signal.filtfilt(numerator, denominator, data)
    return output


def moving_average(values, window, shift=True):
    """
    Creates a pandas Series as moving average of the input series.

    :param values: Series
        For dataframe e.g. df['a_col_name'].values
    :param window: int
        sample rate of input
    :param shift: Boolean
        if True, shift array back by window/2 and fill up values at start and end
    :return: numpy.array
        shape has (###,). First and last points of input Series are extrapolated as constant
        values (hold first and last point).
    """
    # TODO How to implement the shift parameter
    window = int(window)
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, 'valid')
    # Create array with first entries and window/2 elements
    fill_start = np.full((int(np.floor(window/2)), 1), sma[0])
    # Same with last value of -values-
    fill_end = np.full((int(np.ceil(window/2)), 1), sma[-1])
    # Stack the arrays
    sma = np.concatenate((fill_start[:, 0], sma, fill_end[:, 0]), axis=0)
    return sma


def create_on_off_signal(df, col_names, threshold, col_names_new):
    """
    Create on and off signals based on the given threshold for all column names.
    :param df: pd.DataFame
        DataFrame with the data to process
    :param col_names: list
        Column names to convert to signals
    :param threshold: float, list of floats
        Threshold for all column-names (single float) or
        a list with specific thresholds for specific columns.
    :param col_names_new: list
        New name for the signal-column
    :return: pd.DataFrame
        Now with the created signals.
    """
    if len(col_names) != len(col_names_new):
        raise IndexError("Given lists differ in length. col_names: {}, "
                         "col_names_new: {}".format(len(col_names), len(col_names_new)))
    if isinstance(threshold, list):
        if len(col_names) != len(threshold):
            raise IndexError("Given lists differ in length. col_names: {}, "
                             "threshold: {}".format(len(col_names), len(threshold)))
    else:
        threshold = [threshold for _ in enumerate(col_names)]
    # Do on_off signal creation for all desired columns
    for i, _ in enumerate(col_names):
        # Create zero-array
        df[col_names_new[i]] = 0.0
        # Change all values to 1.0 according to threshold
        df.loc[df[col_names[i]] >= threshold[i], col_names_new[i]] = 1.0
    return df


def number_lines_totally_na(df):
    """
    Returns the number of rows in the given dataframe
    that are filled with NaN-values.
    :param df: pd.DataFrame
        Given dataframe to process
    :return: int
        Number of NaN-Rows.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas data frame')
    counter = 0
    for _, row in df.iterrows():
        # Check if the whole row is filled with NaNs.
        if all(row.isnull()):
            counter += 1
    return counter


def z_score(x, limit=3):
    """
    Calculate the z-score using the mea
    and standard deviation of the given data.
    :param x: np.array
        For dataframe e.g. df['a_col_name'].values
    :param limit: float, default 3
        Lower limit for required z-score
    :return: iqr: np.array
        modified z score"""
    mean = np.mean(x)
    standard_deviation = np.std(x)
    z_score_value = (x-mean)/standard_deviation
    return np.where(np.abs(z_score_value) > limit)[0]


def modified_z_score(x, limit=3.5):
    """
    Calculate the modified z-score using the median
    and median average deviation of the given data.
    :param x: np.array
        For dataframe e.g. df['a_col_name'].values
    :param limit: float, default 3.5
        Lower limit for required z-score
    :return: iqr: np.array
        modified z score
    """
    median = np.median(x)
    median_average_deviation = np.median(np.abs(x-median))
    z_score_mod = 0.6745*(x-median)/median_average_deviation
    return np.where(np.abs(z_score_mod) > limit)[0]


def interquartile_range(x):
    """
    Calculate interquartile range of given array.
    :param x: np.array
        For dataframe e.g. df['a_col_name'].values
    :return: iqr: np.array
        Array matching the interquartile-range
    """
    quartile_1, quartile_3 = np.percentile(x, [25, 75])
    iqr = quartile_3 - quartile_1
    lower = quartile_1 - (iqr * 1.5)
    upper = quartile_3 + (iqr * 1.5)
    return np.where((x > upper) | (x < lower))[0]
