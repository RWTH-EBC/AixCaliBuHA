# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 18:13:41 2018

@author: tst-aes
revised by:
* 2018-09-24 Philipp Mehrfeld

This is a script to perform supervised learning and apply this to time series data and characterize classes for
these time series.
"""
import os
import matplotlib as mpl
mpl.use('Qt5Agg')
import pylab as plt
import pandas as pd
import pydot
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
# Download binaries from: http://www.graphviz.org/download/
# Windows Zip file directly from: https://graphviz.gitlab.io/_pages/Download/Download_windows.html
os.environ["PATH"] += os.pathsep + (r'D:\06_Programme\graphviz-2.38\release\bin').replace('\\', '/')

def test_creat_and():
    pass

def create_and_export_decision_tree(col_name_list, X_train, y_train, save_path):
    # Create tree instance. Ordne bekannte Klassen den bekannten Trainingsdaten zu
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)

    # Visualization decision tree
    dot_data = StringIO()
    export_graphviz(dtree, out_file=dot_data, feature_names=col_name_list, filled=True, rounded=True)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    Image(graph[0].create_png())  # Creating the image needs some time
    plt.show(graph[0])
    graph[0].write_png(save_path+'/tree_plot.png')

    return dtree


def create_and_report_classification_predictions(dtree, X_test, y_test, df, classes_name, save_path):
    # Here classification report function begins
    # Read in test data file (if not already splitted in the data set above)

    # Predict classes for test data set
    predictions = dtree.predict(X_test)

    # Compare know classes with predicted classes (only possible wenn vorher manuell zugeordnet)
    # TODO Als Export-File!!
    print(classification_report(y_test, predictions))
    print('\n')
    print(confusion_matrix(y_test, predictions))

    # Visualization pair plot (df is data frame with whole X values (train and test)
    sns.pairplot(df, hue=classes_name)  # This function takes a long time to be executed
    plt.savefig(save_path+'/pairplot.png', bbox_inches='tight', dpi=400)

    return plt.gcf()


def main():
    # Define main inputs
    #fname_input = os.path.normpath(r'D:\04_Git\modelica-calibration\Classifier\ClassifierInput.xlsx')  # file
    #model_input = pd.read_excel(io=fname_input, sheet_name='Sheet1')  # Index of data frame is first column. However, index is not important in this function
    #col_name_list = ['VDot', 'T_RL', 'T_VL', 'T_Amb', 'MassFlow', 'TempDiff']  # List with column names that should be part of the classifier analysis
    #col_name_with_classes = 'class'  # Column name where classes are listed

    fname_input = os.path.normpath(r'D:\CalibrationHP\2018-01-30\AllData_RowDiv10_Interp10s.hdf')
    model_input = pd.read_hdf(fname_input)
    col_name_list = []  # List with column names that should be part of the classifier analysis
    col_name_with_classes = ''  # Column name where classes are listed

    save_path = os.path.normpath(r'D:')  # Result folder

    X = model_input[col_name_list].copy()  # Data frame with interesting vaules
    y = model_input[col_name_with_classes].copy()  # Copy column with class descriptions and distribution to new Series

    # Split data set randomly with test_size % (if 0.30 --> 70 % are training data)
    # TODO However specifying yourself which data set is for training and which for testing should also be implemented!
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    my_dtree = create_and_export_decision_tree(col_name_list=col_name_list, X_train=X_train,
                                               y_train=y_train, save_path=save_path)
    my_fig = create_and_report_classification_predictions(dtree=my_dtree, X_test=X_test, y_test=y_test,
                                                          df=model_input[col_name_list+[col_name_with_classes]],
                                                          classes_name=col_name_with_classes, save_path=save_path)
    my_fig.show()


if __name__=='__main__':
    main()