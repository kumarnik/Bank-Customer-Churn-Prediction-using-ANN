# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt


#reading data
data=pd.read_csv("C:\\Users\\nikhil\\Desktop\\churn_data\\churn-modellingcsv\\Churn_Modelling.csv")

x=data.iloc[:,3:13]
y=data.iloc[:,13]

#finding any null value in the dataset
find_null=data.isnull().sum()

# handle categorical variables:
# creating categorical variable
geography=pd.get_dummies(x['Geography'],drop_first=True)

gender=pd.get_dummies(x['Gender'],drop_first=True)

#concat the new variables into the dataset
X=pd.concat([x,geography,gender],axis=1)

#drop column which are redundant and of no use
X=X.drop(['Geography','Gender'],axis=1)


#Split data into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


#feature scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

 #Modelling:
 
 #SVM:
from sklearn.svm import SVC
svm=SVC()
model_svm=svm.fit(X_train,y_train)
pred_y_svm=model_svm.predict(X_test)

#calculating the accuracy and confusion matrix
from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_svm=accuracy_score(pred_y_svm,y_test)
conf_svm=confusion_matrix(pred_y_svm,y_test)

#Random-Forest:
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=20)
model_rf=rf.fit(X_train,y_train)
pred_y_rf=model_rf.predict(X_test)


accuracy_rf=accuracy_score(pred_y_rf,y_test)
conf_rf=confusion_matrix(pred_y_rf,y_test)

#import keras library and packages:
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialise the ANN:
classifier=Sequential()

#ADD input layer and 1st hidden layer
classifier.add(Dense(input_dim=11, units=5,activation="sigmoid",kernel_initializer="uniform"))

#added the output layer
classifier.add(Dense(units=1,activation="sigmoid",kernel_initializer="uniform"))

#compiling the ANN:
classifier.compile(optimizer="sgd",loss="binary_crossentropy")

#Fitting the ANN to the training dataset
model_ann=classifier.fit(X_train,y_train,batch_size=50,epochs=50)


#Making Predictions and accuracy of the model:
pred_y_ann=classifier.predict(X_test)
pred_y_ann=(pred_y_ann>0.5)


#finding accuracy and confusion-matrix:
from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_ann=accuracy_score(pred_y_ann,y_test)   #79.5666
conf_ann=confusion_matrix(pred_y_ann,y_test)

#plot the loss varies corresponding to the epochs:
plt.plot(model_ann.history['loss'])
#plt.plot(model_ann.history['accuracy'])




#2nd ANN: for different parameter values:
#like activation function:relu
classifier2=Sequential()
classifier2.add(Dense(input_dim=11, units=5,activation="relu",kernel_initializer="uniform"))

classifier2.add(Dense(units=1,activation="sigmoid",kernel_initializer="uniform"))
classifier2.compile(optimizer="sgd",loss="binary_crossentropy",metrics=['accuracy'])


model_ann2=classifier2.fit(X_train,y_train,batch_size=50,epochs=50)

pred_y_ann2=classifier2.predict(X_test)
pred_y_ann2=(pred_y_ann2>0.5)

from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_ann2=accuracy_score(pred_y_ann2,y_test)   
conf_ann2=confusion_matrix(pred_y_ann2,y_test)

plt.plot(model_ann2.history['loss'])
plt.plot(model_ann2.history['accuracy'])



#3rd ANN:
# ADDED one more hidden layer:
classifier3=Sequential()

#input and first hidden layer:
classifier3.add(Dense(input_dim=11, units=6,activation="relu",kernel_initializer="uniform"))

#2nd hidden layer
classifier3.add(Dense(units=5,activation="relu",kernel_initializer="uniform"))


#output layer
classifier3.add(Dense(units=1,activation="sigmoid",kernel_initializer="uniform"))
classifier3.compile(optimizer="sgd",loss="binary_crossentropy",metrics=['accuracy'])


model_ann3=classifier3.fit(X_train,y_train,batch_size=50,epochs=50)

pred_y_ann3=classifier3.predict(X_test)
pred_y_ann3=(pred_y_ann3>0.5)

from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_ann3=accuracy_score(pred_y_ann3,y_test)   
conf_ann3=confusion_matrix(pred_y_ann3,y_test)

plt.plot(model_ann3.history['loss'])
plt.plot(model_ann3.history['accuracy'])



#4th ANN:
classifier4=Sequential()
classifier4.add(Dense(input_dim=11, units=6,activation="relu",kernel_initializer="he_uniform"))

classifier4.add(Dense(units=5,activation="relu",kernel_initializer="he_uniform"))

classifier4.add(Dense(units=1,activation="sigmoid",kernel_initializer="uniform"))
classifier4.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])


model_ann4=classifier4.fit(X_train,y_train,batch_size=50,epochs=50)

pred_y_ann4=classifier4.predict(X_test)
pred_y_ann4=(pred_y_ann4>0.5)

from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_ann4=accuracy_score(pred_y_ann4,y_test)   
conf_ann4=confusion_matrix(pred_y_ann4,y_test)

#plotting the loss and accuracy graphs corresponding to the epochs:
plt.plot(model_ann4.history['loss'])
plt.plot(model_ann4.history['accuracy'])














#
#import keras