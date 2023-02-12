import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sbn 


dataFrame=pd.read_csv("merc.csv")
#head(number) prints first number times row in the frame
#print(dataFrame.head(10))
#verileri açıklar
#print(dataFrame.describe())
#boş verileri gösterir
  #print(dataFrame.isnull().sum())
#plt.figure(figsize=(7,5))
   #sbn.displot(dataFrame["price"])
#yıla göre cubuk grafiğini saydı
  #sbn.countplot(x=dataFrame["year"])
#fiyat için korelasyonu hesaplar sort_values sıralama yapar
    #print(dataFrame.corr()["price"].sort_values())
#noktasal grafik verir
#sbn.scatterplot(x="mileage",y="price",data=dataFrame)
#en yüksek fiyatlı ilk 20 aracı gösteriyor
#print(dataFrame.sort_values("price",ascending=False).head(20))
#%0.01 i kaç yapar
#print(len(dataFrame)*0.01)
yenidf=dataFrame.sort_values("price",ascending=False).iloc[131:]
#print(yenidf)
#yeni grafik
#plt.figure(figsize=(7,5))
#print(sbn.distplot(yenidf["price"]))
#yıllara göre fiyatların ortlaması
#print(yenidf.groupby("year").mean()["price"])

#1970 i isteğe bağlı olarak çıkartabiliriz
#print(dataFrame[dataFrame.year!=1970].groupby("year").mean()["price"])
dataFrame=yenidf
dataFrame[dataFrame.year!=1970]
#print(dataFrame.groupby("year").mean()["price"])
dataFrame=dataFrame.drop("transmission",axis=1)

y=dataFrame["price"].values
x=dataFrame.drop(["price","model","fuelType"],axis=1).values

#t train test y train test gibi xy dizilerini bölüyor

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=10)
#print(len( x_test))
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
from keras.models import Sequential 
from keras.layers import Dense
model=Sequential()
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))
model.add(Dense(12,activation="relu"))

model.add(Dense(1))
model.compile(optimizer="adam",loss="mse")
model.fit(x=x_train,y=y_train,validation_data=(x_test,y_test),batch_size=250,epochs=300)
kayıpVerisi=pd.DataFrame(model.history.history)
kayıpVerisi.plot()
plt.show()
from sklearn.metrics import mean_absolute_error,mean_squared_error
tahmindizisi=model.predict(x_test)
#gerçekle tahmin arasındaki fiyat farkını yansıtır
#print(mean_absolute_error(y_test,tahmindizisi))

 