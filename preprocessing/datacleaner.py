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


def simple_plot(df, title, x, y):
    plt.scatter(df['timestamp'], df['value'], color='b')
    # labels = [dt.datetime.fromtimestamp(ele[0]/1000).strftime('%Y-%m-%d %H:%M:%S') for ele in df['timestamp']]
    # plt.xticks(df['timestamp'], np.arange(len(df)), rotation=45)
    plt.title(title, fontsize=20)
    plt.xlabel(x, fontsize=14)
    plt.ylabel(y, fontsize=14)
    plt.grid()
    plt.gcf().autofmt_xdate()
    plt.show()


if __name__=='__main__':
    # The original data
    df = fromJSONDumpFile(r'D:\test\data_diris.csv')
    #TODO nur listeneinträge mit zulässigen Werten zulassen
    desired_freq = '10Min'


    # ZWISCHENTUECK START
    # Etwas ergaenzen
    df_temp = df.iloc[0:8]
    extra_temp = df['value'].copy()
    for i in range(0, len(extra_temp)):
        extra_temp[i] = i*5.0
    df_temp['extra'] = extra_temp
    df = df_temp.copy()
    # ??? Wie sieht es aus, wenn das ende nicht genau passt zu start + X*desired_freq?

    # if reset is desired
    df = df_temp.copy()
    df.drop(columns='timestamp', inplace=True)
    # ZWISCHENTUECK ENDE

    print('Number of invalid values in data frame rows: \n' + str(df.isnull().sum()))  # Count invalid values
    # Count rows  where all fields are invalid values
    counter_temp = 0
    row_len_temp = len(df.columns)  # Calculate number of columns such that statement is not executed in evey for execution
    for row in range(len(df.index)):
        if df.iloc[row].isnull().sum() >= row_len_temp:
            counter_temp += 1
    print('Number of rows in data frame where the whole line consists of invalid values: ' + str(counter_temp))
    del counter_temp, row_len_temp
    df.dropna(inplace=True, axis=0)  # Drop all rows where at least one NaN exists
    # TODO Vielleicht optional ergänzen, dass nur die Zeilen gelöscht werden sollen, die komplett NaN sind.
    # TODO Zeilen mit doppelten oder beliebig mehrfachem Indexeintrag müssen zu einem zusammengeführt werden:
    # z. B. index: [2017-03-12 23:31:35, 2017-03-12 23:31:35, 2017-03-12 23:31:35]
    # value: [13.0, 15.32, 23.94] müssen werden zu:
    # index = [2017-03-12 23:31:35] und wert = [17,42] (da (13+15.32+23.94) / 3)
    # etwas wie das folgende könnte hilfreich sein danach: df.drop_duplicates(subset='timestamp', keep='last')
    # TODO vorher unbedingt den index vom orig df zum typ pandas.core.indexes.datetimes.DatetimeIndex konvertieren
    time_index = pd.date_range(start=df.index[0], end=df.index[-1], freq=desired_freq)  # Create new equally spaced index
    df_time_temp = pd.DataFrame(index=time_index)  # Create in empty data frame
    df = pd.concat([df, df_time_temp], axis=1)  # Concatenate two data frames. Actually only new index is inserted
    del df_time_temp
    df.interpolate(method='time', axis=0, inplace=True)  # Interpolate linearly according to time index
    df = df.resample(rule=desired_freq).mean()  # Resample to equally spaced index.
                                                # All fields should already have a value. Thus NaNs and maybe +/- infs
                                                # should have been filtered beforehand.


    print(df)
    simple_plot(df, "Measurement", "Time", "Value(Watt)")
