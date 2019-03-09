import tensorflow as tf
import numpy as np

N=300
ND=1000
Noise_Amp=.1
slope=np.random.rand(ND)
dataset=[]
for value in slope:
    noise=np.random.normal(0,Noise_Amp,N)
    x=np.linspace(0,value,N)
    dataset.append(x+noise)
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
              metrics=['MSE','MAE','accuracy'])
#model.summary()
model.fit(x_train, y_train, epochs=100, batch_size=50, verbose=0,validation_split=.2)

model.evaluate(x_test, y_test)

slope_cal=np.linspace(0,.9,10)
dataset_cal=[]

for value in slope_cal:
    dataset_cal.append(np.linspace(0,value,N)+np.random.normal(0,Noise_Amp,N))

x_cal=np.asarray(dataset_cal)

print('if working',slope_cal.reshape(-1,1),'will be the same as',model.predict(x_cal))


    #dataset.append(x)

