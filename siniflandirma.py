import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sbn 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.callbacks import EarlyStopping
dataFrame=pd.read_excel("maliciousornot.xlsx")
#print(dataFrame.corr()["Type"].sort_values())
#sbn.countplot(x="Type",data=dataFrame)
dataFrame.corr()["Type"].sort_values().plot(kind="bar")
y=dataFrame["Type"].values
x=dataFrame.drop("Type",axis=1).values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=15)
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
model=Sequential()
model.add(Dense(units=30,activation="relu"))
model.add(Dense(units=15,activation="relu"))
model.add(Dense(units=15,activation="relu"))
model.add(Dense(units=1,activation="sigmoid"))
model.compile(loss="binary_crossentropy",optimizer="adam")
model.fit(x=x_train,y=y_train,epochs=700,validation_data=(x_test,y_test),verbose=1)
modelKaybı=pd.DataFrame
plt.show()