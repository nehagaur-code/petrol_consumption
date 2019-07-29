import pandas as pd
import numpy as np

#input file to read data from csv
dataset=pd.read_csv("./petrol_consumption.csv")

X = dataset.drop('Petrol_Consumption', axis=1)  

y = dataset['Petrol_Consumption']  


#split data into Train and Test with the ratio of 80% as train datra and 20 % as Test Data
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  


#import Decision Tree Classifier and fit tree.
from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier()  
classifier.fit(X_train, y_train)  

#predict test data with  the model trained
y_pred = classifier.predict(X_test)  


#imported classification report as well as confusion matrix to visualize the results 
from sklearn.metrics import classification_report, confusion_matrix  

#confusion matrix usage to evaluate the quality of the output of a classifier on the data set.
confusion_matrix=confusion_matrix(y_test, y_pred) 

classification_report=classification_report(y_test, y_pred) 

df_predicted=pd.DataFrame({'Actual_Data':y_test, 'Predicted_Data':y_pred})  

