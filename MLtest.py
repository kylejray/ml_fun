import tensorflow as tf
import numpy as np

np.random.seed(420)
N=10
ND=10
Noise_Amp=.1
slope=np.random.rand(ND)
dataset=[]
for value in slope:
    noise=np.random.normal(0,Noise_Amp,N)
    y=np.linspace(0,value,N)
    #x=y/value
    dataset.append(y+noise)
    #dataset.append(x)


    
(x_train,y_train)=(np.asarray(dataset[0:int(ND*.8)]),slope[0:int(ND*.8)])
(x_test,y_test)=(np.asarray(dataset[int(ND*.8):]),slope[int(ND*.8):])


model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(50, activation=tf.nn.relu,input_dim=N),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(10,activation='relu'),
  tf.keras.layers.Dense(1, activation='linear')
])
model.compile(optimizer='adam',
              loss='MSE',
              metrics=['MAE'])
#model.summary()
#model.fit(x_train, y_train, epochs=500, batch_size=100, verbose=0,validation_split=.2)
model.fit(x_train, y_train, epochs=10, batch_size=50, verbose=0)

result=model.evaluate(x_test, y_test,verbose=1)
print(result)

#slope_cal=np.linspace(0,.9,10)
#dataset_cal=[]

#for value in slope_cal:
    #dataset_cal.append(np.linspace(0,value,N)+np.random.normal(0,Noise_Amp,N))

#x_cal=np.asarray(dataset_cal)

#print('if working',slope_cal.reshape(-1,1),'will be the same as',model.predict(x_cal))


    #dataset.append(x)

