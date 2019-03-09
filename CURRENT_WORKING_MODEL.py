
import numpy as np


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

N=300
ND=100
Noise_Amp=.1
slope=np.random.rand(ND)
dataset=[]
for value in slope:
    noise=np.random.normal(0,Noise_Amp,N)
    x=np.linspace(0,value,N)
    dataset.append(x+noise)
    #dataset.append(x)


    
(X_train,Y_train)=(np.asarray(dataset[0:int(ND*.8)]),slope[0:int(ND*.8)])
(X_test,Y_test)=(np.asarray(dataset[int(ND*.8):]),slope[int(ND*.8):])



model = Sequential()
model.add(Dense(50, input_dim=N, kernel_initializer='normal', activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

history = model.fit(X_train, Y_train, epochs=20,batch_size=50,verbose=1)

slope_cal=np.linspace(0,1,10)
dataset_cal=[]

for value in slope_cal:
    dataset_cal.append(np.linspace(0,value,N)+np.random.normal(0,Noise_Amp,N))

x_cal=np.asarray(dataset_cal)

print('if working',slope_cal.reshape(-1,1),'will be the same as',model.predict(x_cal))


