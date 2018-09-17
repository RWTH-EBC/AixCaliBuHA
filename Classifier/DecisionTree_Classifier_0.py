# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 18:13:41 2018

@author: tst-aes

model_input is a pandas DataFrame with time series data in the columns (col 1: time, 2: inlet temp, 3: outlet temp, 4: volume flow rate,  ...)
"""

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


sns.pairplot(model_input,hue='class')
plt.savefig('D:\Asad\Sim\SimResults\Plots\Pairplot_'+ str(StartRange)+'_'+ str(EndRange) + '.png',transparent=True,bbox_inches='tight',dpi=400)