import os
import pandas as pd

file = os.path.normpath(r'D:\01_Dymola_WorkDir\00_Testzone\Unnamed_with_events.hdf')  # event at 2.34 s
my_df_events = pd.read_hdf(file)

#2000-01-01 00:00:01.000   3.100    ...            1.0
#2000-01-01 00:00:02.000   6.200    ...            1.0
#2000-01-01 00:00:02.340   7.254    ...            1.0
#2000-01-01 00:00:02.340   5.850    ...            1.0
#2000-01-01 00:00:03.000   7.500    ...            1.0

from ebccalibration.preprocessing.datacleaner import clean_and_space_equally_time_series

neues_df = clean_and_space_equally_time_series(df=my_df_events, desired_freq='0.4S')