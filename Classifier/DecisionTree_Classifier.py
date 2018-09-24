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
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz

# Download binaries from: http://www.graphviz.org/download/
# Windows Zip file directly from: https://graphviz.gitlab.io/_pages/Download/Download_windows.html
os.environ["PATH"] += os.pathsep + (r'D:\06_Programme\graphviz-2.38\release\bin').replace('\\', '/')

# Define main inputs
# clases: 0 - Cool down when boiler off; 1 - Operation in hysteresis mode;
# 2 - Heat up phase (starting phase); 7 - Certain error
fname_input = (r'D:\04_Git\modelica-calibration\Classifier\ClassifierInput.xlsx').replace('\\', '/') # file
save_path_plots = (r'D:').replace('\\', '/') # folder
model_input = pd.read_excel(io=fname_input, sheet_name='Sheet1')
StartRange = 0
EndRange = 1672
start_col = 1  # In example file this is column "C: VDot"
end_col = 8  # In example file this is column "I: TempDiff"

X = model_input.drop('class', axis=1)
X = X.drop('time', axis=1)
y = model_input['class']

# Split data set randomly with test_size % (if 0.30 --> 70 % are training data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Create tree instance. Ordne bekannte Klassen den bekannten Trainingsdaten zu
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

# Visualization decision tree
features = list(model_input.columns[start_col:end_col])
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data, feature_names=features, filled=True, rounded=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
Image(graph[0].create_png())
plt.show(graph[0])
graph[0].write_png(save_path_plots+'/tree_plot.png')

# Read in test data file

# Predict classes for test data set
predictions = dtree.predict(X_test)

# Compare know classes with predicted classes (only possible wenn vorher manuell zugeordnet)
print(classification_report(y_test, predictions))
print('\n')
print(confusion_matrix(y_test, predictions))

# Visualization pair plot
sns.pairplot(model_input, hue='class')
plt.savefig(save_path_plots+'/pairplot_' + str(StartRange) + '_' + str(EndRange) + '.png', transparent=True, bbox_inches='tight',
            dpi=400)
