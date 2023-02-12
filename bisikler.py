import pandas as pd 
import seaborn as sbn
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_absolute_error,mean_squared_error

dataframe=pd.read_excel("bisiklet_fiyatlari.xlsx")

sbn.pairplot(dataframe)

#veriyi test/train olarak ikiye ayırma
#y=wx+b 
#y -> label
#x -> feature
y=dataframe["Fiyat"].values
x=dataframe[["BisikletOzellik1","BisikletOzellik2"]].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=15)
"""
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
"""
#scaling boyutunu değiştirmek
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)


model=Sequential()
model.add(Dense(4,activation="relu"))
model.add(Dense(4,activation="relu"))
model.add(Dense(4,activation="relu"))
model.add(Dense(1))
model.compile(optimizer="rmsprop",loss="mse")
model.fit(x_train,y_train,epochs=250)

#loss=model.history.history["loss"]
#sbn.lineplot(x=range(len(loss)),y=loss)

trainLoss=model.evaluate(x_train,y_train,verbose=0)
testLoss=model.evaluate(x_test,y_test,verbose=0)

test_tahmin=model.predict(x_test)

test_tahmin=pd.Series(test_tahmin.reshape(330,))
tahminDf=pd.DataFrame(y_test,columns=["gerçek Y"])
tahminDf=pd.concat([tahminDf,test_tahmin],axis=1)
tahminDf.columns=["gerçek y","tahmin y"]
print(tahminDf)
sbn.scatterplot(x="gerçek y",y="tahmin y",data=tahminDf)

print(mean_absolute_error(tahminDf["gerçek y"],tahminDf["tahmin y"]))
