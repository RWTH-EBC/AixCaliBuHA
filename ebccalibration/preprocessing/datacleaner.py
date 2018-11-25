import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import pandas as pd
import json
import sys
#sys.path.append('Datareader')
#import paDataReader as paDataReader

def fromJSONDumpFile (path):
    """
    Read data from a csv file containing data in the following format:
        "[timestamp, value]", "timestamp, value]", ...
    """
    # Read CSV, transpose the DataFrame and decode each json array
    df = pd.read_csv(path, header=None).T.applymap(json.loads)
    # make a new DataFrame with separate columns
    df2 = pd.DataFrame(df[0].values.tolist(), columns=["timestamp", "value"])
    # add timestamp also as index
    df2.index = pd.to_datetime(df2['timestamp'], unit='ms')
    return df2


def number_lines_totally_na(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas data frame')
    counter = 0
    row_len = len(df.columns)  # Calculate number of columns such that statement is not executed in evey for execution
    for row in range(len(df.index)):
        if df.iloc[row].isnull().sum() >= row_len:
            counter += 1
    return counter


def clean_and_space_equally_time_series(df, desired_freq):
    # TODO doc_strings !!!
    # TODO nur listeneinträge mit zulässigen Werten (z. B. alles Zahlenwerte bzw. index korrekter type) zulassen
    # TODO vorher unbedingt den index vom orig df zum typ pandas.core.indexes.datetimes.DatetimeIndex konvertieren

    # 1 (start): Check dataframe for NANs
    # Create a pandas Series with number of invalid values for each column of df
    series_with_na = df.isnull().sum()
    for name in series_with_na.index:
        if series_with_na.loc[name] > 0:
            print(name + ' has following number of invalid values\n' + str(series_with_na.loc[name]))  # Print only columns with invalid values
    # Drop all rows where at least one NA exists
    df.dropna(how='any', inplace=True)
    # 1 (end)

    # 2 (start): dataframe rows (indices) that exist multiple times will be identified and set to their mean value. e.g.:
    # 2000-01-01 00:00:02.000   6.200    ...            1.0
    # 2000-01-01 00:00:02.340   6.500    ...            1.0
    # 2000-01-01 00:00:02.340   7.000    ...            1.0
    # 2000-01-01 00:00:03.000   7.500    ...            1.0
    # wil become:
    # 2000-01-01 00:00:02.000   6.200    ...            1.0
    # 2000-01-01 00:00:02.340   6.750    ...            1.0
    # 2000-01-01 00:00:03.000   7.500    ...            1.0
    double_ind = df.index[df.index.duplicated()].unique() # Find entries that are exactly the same timestamp
    mean_values = []
    for item in double_ind:
        #array_temp = df.index.get_loc(item)  # Must be type slice (see examples in doc of pandas get_loc)
        mean_values.append(df.loc[item].values.mean(axis=0))
    df = df[~df.index.duplicated(keep='first')].copy()  # Delete duplicate indices
    # Set mean values in rows that were duplicates before
    for idx, values in zip(double_ind, mean_values):
        df.loc[idx] = values
    # 2 (end)


    # 3 (start): resampling to new frequency with linear interpolation
    # Create new equally spaced DatetimeIndex. Last entry is always < df.index[-1]
    time_index = pd.date_range(start=df.index[0], end=df.index[-1], freq=desired_freq)
    df_time_temp = pd.DataFrame(index=time_index)  # Create in empty data frame
    # concat and radd might be used as alternatives here
    #df = pd.concat([df, df_time_temp], axis=1)  # Concatenate two data frames. Actually only new index is inserted
    df = df.radd(df_time_temp, axis='index', fill_value=0)  # Insert temporary time_index into df. fill_value = 0 can only be used, since all NaNs should be eliminated prior.
    del df_time_temp
    df.interpolate(method='time', axis=0, inplace=True)  # Interpolate linearly according to time index
    # Determine Timedelta between current first index entry in df and the first index entry that would be created
    # when applying df.resample() without loffset
    delta_time = df.index[0] - df.resample(rule=desired_freq).first().first(desired_freq).index[0]
    df = df.resample(rule=desired_freq, loffset=delta_time).first()  # Resample to equally spaced index.
                                                # All fields should already have a value. Thus NaNs and maybe +/- infs
                                                # should have been filtered beforehand.
    del delta_time
    # 3 (end)

    return df


def create_on_off_acc_to_threshold(df, col_names, threshold, names_new_columns):
    for i in range(len(col_names)):  # Do on_off signal creation for all desired columns
        df[names_new_columns[i]] = df[col_names[i]].copy()  # Copy column with values to new one, where only zeros and ones will exist.
        df.loc[df[names_new_columns[i]] < threshold, names_new_columns[i]] = 0.0
        df.loc[df[names_new_columns[i]] >= threshold, names_new_columns[i]] = 1.0
    return df


def movingaverage(values, window, shift=True):
    '''
    Creates a pandas Series as moving average of the input series.

    :param values: Series
    :param window: int (sample rate of input)
    :param shift: Boolean
        if True, shift array back by window/2 and fill up values at start and end
    :return: numpy.array
        shape has (###,). First and last points of input Series are extroplated as constant
        values (hold first and last point).
    '''
    window = int(window)
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, 'valid')
    fill_start = np.full((int(np.floor(window/2)), 1), sma[0])  # Create array with first entries and window/2 elements
    fill_end = np.full((int(np.ceil(window/2)), 1), sma[-1])  # Same with last value of -values-
    sma = np.concatenate((fill_start[:,0], sma, fill_end[:,0]), axis=0)  # Stack the arrays
    return sma


def low_pass_filter(input, crit_freq, filter_order):
    '''
    :param input: numpy.ndarray,
        For dataframe e.g. df['a_col_name'].values
    :param crit_freq: float,
    :param filter_order: int,
    :return: numpy.ndarray,
    '''
    b, a = signal.butter(N=filter_order, Wn=crit_freq, btype='low', analog=False, output='ba')
    output = signal.filtfilt(b, a, input)
    return output


def zScore(x):
    mean = np.mean(x)
    sd = np.std(x)
    zscore = [(i - mean) / sd for i in x]
    return np.where(np.abs(zscore) > 3)[0]


def zScoreMod(x):
    median = np.median(x)
    mad = np.median([np.abs(i - median) for i in x])
    zscoremod = [0.6745 * (i - median) / mad for i in x]
    return np.where(np.abs(zscoremod) > 3.5)[0]


def iqr(x):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lower = q1 - (iqr * 1.5)
    upper = q3 + (iqr * 1.5)
    return np.where((x > upper) | (x < lower))[0]


if __name__=='__main__':
    # The original data
    #df = fromJSONDumpFile(r'D:\test\data_diris.csv')
    # Other data
    import os
    fname_input = os.path.normpath(r'D:\CalibrationHP\2018-01-26\AllData_orig_10s.hdf')
    df = pd.read_hdf(fname_input)


    # Only for testing. Creates smaller sub-dataframe
    #names = ['HYD.Bank[4].Temp_extRL.REAL_VAR','HYD.Bank[4].Temp_extVL.REAL_VAR','KK.Leistungsmessung_L1_30A.REAL_VAR']
    #rng2 = pd.date_range(start=df.index[2], end=df.index[2], periods=3)
    #rng3 = pd.date_range(start=df.index[5], end=df.index[5], periods=2)
    #df = df.head(9)#.copy()
    #df = df[names]#.copy()
    #rng = df.index[0:2].append(rng2).append(rng3).append(df.index[6:-1])
    #df.set_index(rng, drop=True, inplace=True)


    # Define name of variable to plot which should be the name of the data frame column header
    plot_var = 'value'
    plot_var = 'KK.Leistungsmessung_L1_30A.REAL_VAR'

    clean_df = clean_and_space_equally_time_series(df=df, desired_freq='9S')  # Or .iloc[0:100]; 10Min for JSON input

    #print(df)
    #print(clean_df)

    # Plotting Start
    my_fig, my_ax = plt.subplots(nrows=1, ncols=1)
    my_ax.scatter(clean_df.index, clean_df[plot_var], color='b')
    my_ax.set_title('Measurement')
    my_ax.set_xlabel('Date Time'); my_ax.set_ylabel(plot_var)
    time_span = clean_df.index[-1] - clean_df.index[0]
    my_ax.set_xlim([clean_df.index[0]-0.05*time_span, clean_df.index[-1]+0.05*time_span])  # +/- 5 % of time span
    del time_span
    my_ax.grid(); my_fig.autofmt_xdate()
    my_fig.show()
    # Plotting End


    # Some data filters
    ## Low pass filter. Be careful! Strong negative slopes from e.g. 1000 down to 0 will result in negative values of the filtered output.
    filtered_output = low_pass_filter(input=clean_df[plot_var].values, crit_freq=1/5, filter_order=5)
    ## Moving average. window depends on sample rate of index.
    ## output has less sample points than orig data (len(av_output) = len(orig) - window + 1
    av_output = movingaverage(values=clean_df[plot_var], window=20)
    ## Plot for visualization
    fig, ax = plt.subplots(1,1)
    ax.plot(clean_df[plot_var].values, label='original')
    ax.plot(filtered_output, label='Low Pass')
    ax.plot(av_output, label='Moving Average')
    ax.legend()
    fig.show()


    # Create on off signal according to compressor P_el and a threshold.
    df_with_on_signal = create_on_off_acc_to_threshold(df=clean_df, col_names=['KK.Leistungsmessung_L1_30A.REAL_VAR'], threshold=60, names_new_columns=['HP_on_off'])
    # Save new df
    df_with_on_signal.to_hdf(path_or_buf=os.path.normpath(fname_input)[:-4] + '_with_on_off' + os.path.normpath(fname_input)[-4:], key='df_with_on_off_signal')

    print('ENDE')

    #Jetzt Filter drüber laufen lassen, der alle Werte <= Threshold identifiziert und rauslöscht.
    #Mal schauen. Ich glaube im Rahmen von nxGREASER habe ich das mal gemacht.

    #Tiefpass-filter anbieten