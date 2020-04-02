#Import all library needed for the project.

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#Now import our dataset

data=pd.read_csv('taxi.csv')
print(data.head())

#Splitting input and output data.
data_x=data.drop(['Numberofweeklyriders'],axis=1)
data_y=data['Numberofweeklyriders']
print(data_x.head())
print(data_y.head())

#Splitting the data into training and testing and fit the model. 
X_train,X_test,y_train,y_test = train_test_split(data_x,data_y,test_size=0.3,random_state=0)
reg = LinearRegression()
reg.fit(X_train,y_train)

#Check model accuracy.
print("Train Score:", reg.score(X_train,y_train))
print("Test Score:", reg.score(X_test,y_test))

#save model.
pickle.dump(reg, open('taxi.pkl','wb'))
model = pickle.load(open('taxi.pkl','rb'))
print(model.predict([[80, 1770000, 6000, 85]]))

