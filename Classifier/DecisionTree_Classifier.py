# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 18:13:41 2018

@author: tst-aes
"""
import numpy as np
import pandas as pd
import os
import requests
import graphviz
import sklearn as skl
import pydot
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz

StartRange=0    
EndRange= 1672

X=model_input.drop('class',axis=1)
X=X.drop('time',axis=1)
y=model_input['class']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30)	     
dtree=DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predictions=dtree.predict(X_test)

print (classification_report(y_test,predictions))
print('\n') 
print(confusion_matrix(y_test,predictions))

#Visualization 
features= list(model_input.columns[1:8])
dot_data=StringIO()
export_graphviz(dtree,out_file=dot_data,feature_names=features,filled=True,rounded=True)
graph=pydot.graph_from_dot_data(dot_data.getvalue())
Image(graph[0].create_png())
plt.show(graph[0])

#sns.pairplot(model_input,hue='class')
#plt.savefig('D:\Asad\Sim\SimResults\Plots\Pairplot_'+ str(StartRange)+'_'+ str(EndRange) + '.png',transparent=True,bbox_inches='tight',dpi=400)
