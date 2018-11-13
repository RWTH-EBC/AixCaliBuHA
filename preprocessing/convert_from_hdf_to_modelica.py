import pandas as pd
#import modelicares as mres
import scipy.io as spio

def conv_hdf_to_mat(fname, columns, save_path):
    '''

    :param fname: str,
        String or even os.path.normpath.
        Must point to a valid hdf file.
    :param columns: list,
        A list with names of columns that should be saved to matfile.
    :return mat_file:
         Returns the version 4 matfile
    '''
    df = pd.read_hdf(fname)
    new_mat = {'table': df[columns]}
    spio.savemat(save_path, new_mat, format="4")
    return new_mat


def main():
    import os
    fname_input = os.path.normpath(r'D:\CalibrationHP\2018-01-26\AllData.hdf')
    save_path = os.path.normpath(r'D:\CalibrationHP\2018-01-26')
    my_mat = conv_hdf_to_mat(fname=fname_input, save_path=save_path)


if __name__=='__main__':
    main()