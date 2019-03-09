import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense










N=100
ND=1000
slope=np.random.rand(ND)
dataset=[]
for value in slope:
    noise=np.random.normal(0,.1,N)
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

history = model.fit(X_train, Y_train, epochs=10,batch_size=50,verbose=1)


