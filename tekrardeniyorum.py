import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sbn 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf   
from keras.models import Sequential
from keras.layers import Dense  
from sklearn.metrics import mean_absolute_error,mean_squared_error
dataFrame=pd.read_excel("bisiklet_fiyatlari.xlsx")
"""
print(dataFrame.describe())
sbn.pairplot(dataFrame)
plt.show()"""
# .values ile numpy dizisine cevirebiliyoruz amac x ve y değerlerini test ve train olarak ayırmak 
#x burada özellikler y ise amaç ya da label(işaret) olarak adlandırılıyor
x=dataFrame.drop("Fiyat",axis=1).values
y=dataFrame["Fiyat"].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=15)
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
loss=model.history.history["loss"]

""" modelin cıbuk grafiğini cizer
sbn.lineplot(x=range(len(loss)),y=loss)
plt.show()"""

trainloss=model.evaluate(x_train,y_train,verbose=0)
testloss=model.evaluate(x_test,y_test,verbose=0)

testTahminleri=model.predict(x_test)
gercekDf=pd.DataFrame(y_test,columns=["Gerçek."])
testTahminleri=pd.Series(testTahminleri.reshape(330,))
gercekDf=pd.concat([gercekDf,testTahminleri],axis=1)
gercekDf.columns=["Gerçek","Tahmin"]
print(gercekDf)
""" gerçek tahminin yapılan tahmine göre grafiği çizilebilir
sbn.scatterplot(x="Gerçek",y="Tahmin",data=gercekDf)
plt.show()"""

print(mean_absolute_error(gercekDf["Gerçek"],gercekDf["Tahmin"] ))
print(mean_squared_error(gercekDf["Gerçek"],gercekDf["Tahmin"] ))


yeniBisiklet=[[1760,1758]]
yeniBisiklet=scaler.transform(yeniBisiklet)
print(model.predict(yeniBisiklet))
from keras.models import load_model
#haydetmek için
model.save("bisiklet_modeli.h5")
#modeli cağırmak için
#cagirilanModel=load_model("bisiklet_modeli.h5")








