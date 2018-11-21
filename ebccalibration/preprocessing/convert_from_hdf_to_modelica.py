import pandas as pd
#import modelicares as mres
import scipy.io as spio
#import numpy as np

def conv_hdf_to_mat(fname, columns, save_path_file, set_time_to_zero=True):
    '''

    :param fname: str, os.path.normpath
        String or even os.path.normpath.
        Must point to a valid hdf file.
    :param save_path_file: str, os.path.normpath
        File path and name where to store the output *.mat file.
    :param columns: list,
        A list with names of columns that should be saved to *.mat file.
    :param set_time_to_zero: bool (default True),
        If True, the index, which is the time, will be start at a zero base.
    :return mat_file:
         Returns the version 4 matfile
    '''
    df = pd.read_hdf(fname)
    if set_time_to_zero:
        df.index = df.index - df.iloc[0].name.to_datetime64()  # Make index zero based
    df['time_vector'] = df.index.total_seconds()  # Copy values of index as seconds into new column
    columns = ['time_vector'] + columns  # Add column name of time in front of desired columns, since time must be first column in the *.mat file.
    subset = df[columns]  # Store desired columns in new variable, which is a np.array
    new_mat = {'table': subset.values.tolist()}  # Convert np.array into a list and create a dict with 'table' as matrix name
    spio.savemat(save_path_file, new_mat, format="4")  # Save matrix as a MATLAB *.mat file, which is readable by Modelica.

    return new_mat


def main():
    import os
    fname_input = os.path.normpath(r'D:\CalibrationHP\2018-01-26\AllData_with_on_off.hdf')
    save_path = os.path.normpath(r'D:\04_Git\Calibration_And_Analysis_Of_Hybrid_Heat_Pump_Systems\CalibrationModules\Resources\SimulationInputData\CalibrateTzerraHP\2018-01-26_with_on_off.mat')
    meas_variables = [
        'KK.Temp_KK',
        'EXT.Vdot_WP_Lade.REAL_VAR',
        'EXT.T_WP_RL.REAL_VAR',
        'HP_on_off']
    objective_variables = [
        'EXT.T_WP_VL.REAL_VAR',
        'KK.Leistungsmessung_L1_30A.REAL_VAR',
        'KK.Leistungsmessung_L2_30A.REAL_VAR',
        'KK.Leistungsmessung_L3_30A.REAL_VAR']
    my_mat = conv_hdf_to_mat(fname=fname_input, columns=meas_variables+objective_variables, save_path_file=save_path)


if __name__=='__main__':
    main()