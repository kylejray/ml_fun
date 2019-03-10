import tensorflow as tf
import numpy as np
import data_prep

x_train,y_train,x_test,y_test=data_prep.data_prep()

N=len(x_train[0])

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(200, activation=tf.nn.relu,input_dim=N),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#model.summary()
#model.fit(x_train, y_train, epochs=500, batch_size=100, verbose=0,validation_split=.2)
model.fit(x_train, y_train, epochs=20, batch_size=10, verbose=1)

result=model.evaluate(x_test, y_test,verbose=1)
print(result)

#slope_cal=np.linspace(0,.9,10)
#dataset_cal=[] 

#for value in slope_cal:
    #dataset_cal.append(np.linspace(0,value,N)+np.random.normal(0,Noise_Amp,N))

#x_cal=np.asarray(dataset_cal)

#print('if working',slope_cal.reshape(-1,1),'will be the same as',model.predict(x_cal))


    #dataset.append(x)

