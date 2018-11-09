import matplotlib.pyplot as plt
import numpy as np
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

    # Create a pandas Series with number of invalid values for each column of df
    series_with_na = df.isnull().sum()
    for name in series_with_na.index:
        if series_with_na.loc[name] > 0:
            print(name + ' has folling number of invalid values\n' + str(series_with_na.loc[name]))  # Print only columns with invalid values
    # Drop all rows where at least one NA exists
    df.dropna(how='any', inplace=True)
    # Count rows  where all fields are invalid values
    #print('Number of rows in data frame where the whole line consists of invalid values: '
    #      + str(number_lines_totally_na(df)))
    #df.dropna(how='all', inplace=True)  # Drop all rows where at all entries in a row are NA.
    # TODO Zeilen mit doppelten oder beliebig mehrfachem Indexeintrag müssen zu einem zusammengeführt werden:
    # z. B. index: [2017-03-12 23:31:35, 2017-03-12 23:31:35, 2017-03-12 23:31:35]
    # value: [13.0, 15.32, 23.94] müssen werden zu:
    # index = [2017-03-12 23:31:35] und wert = [17,42] (da (13+15.32+23.94) / 3)
    # etwas wie das folgende könnte hilfreich sein danach: df.drop_duplicates(subset='timestamp', keep='last')
    # TODO vorher unbedingt den index vom orig df zum typ pandas.core.indexes.datetimes.DatetimeIndex konvertieren

    # Create new equally spaced DatetimeIndex. Last entry is always < df.index[-1]
    time_index = pd.date_range(start=df.index[0], end=df.index[-1], freq=desired_freq)
    df_time_temp = pd.DataFrame(index=time_index)  # Create in empty data frame
    df = pd.concat([df, df_time_temp], axis=1)  # Concatenate two data frames. Actually only new index is inserted
    del df_time_temp
    df.interpolate(method='time', axis=0, inplace=True)  # Interpolate linearly according to time index
    # Determine Timedelta between current first index entry in df and the first index entry that would be created
    # when applying df.resample() without loffset
    delta_time = df.index[0] - df.resample(rule=desired_freq).mean().first(desired_freq).index[0]
    df = df.resample(rule=desired_freq, loffset=delta_time).mean()  # Resample to equally spaced index.
                                                # All fields should already have a value. Thus NaNs and maybe +/- infs
                                                # should have been filtered beforehand.
    del delta_time
    return df


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
    fname_input = os.path.normpath(r'D:\CalibrationHP\2018-01-26\AllData.hdf')
    df = pd.read_hdf(fname_input)

    # Define name of variable to plot which should be the name of the data frame column header
    plot_var = 'value'
    plot_var = 'KK.Leistungsmessung_L1_30A.REAL_VAR'

    clean_df = clean_and_space_equally_time_series(df=df, desired_freq='1S')  # Or .iloc[0:100]; 10Min for JSON input

    #print(df)
    #print(clean_df)

    my_fig, my_ax = plt.subplots(nrows=1, ncols=1)
    my_ax.scatter(clean_df.index, clean_df[plot_var], color='b')
    my_ax.set_title('Measurement')
    my_ax.set_xlabel('Date Time'); my_ax.set_ylabel(plot_var)
    time_span = clean_df.index[-1] - clean_df.index[0]
    my_ax.set_xlim([clean_df.index[0]-0.05*time_span, clean_df.index[-1]+0.05*time_span])  # +/- 5 % of time span
    del time_span
    my_ax.grid(); my_fig.autofmt_xdate()
    my_fig.show()
    print('ENDE')

    #Jetzt Filter drüber laufen lassen, der alle Werte <= Threshold identifiziert und rauslöscht.
    #Mal schauen. Ich glaube im Rahmen von nxGREASER habe ich das mal gemacht.