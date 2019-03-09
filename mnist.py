import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(mx_train, my_train),(mx_test, my_test) = mnist.load_data()
mx_train, mx_test = mx_train / 255.0, mx_test / 255.0
my_train,my_test=my_train/10,my_test/10

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(100, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1, activation='linear')
])
model.compile(optimizer='adam',
              loss='MSE',
              metrics=['MSE'])
model.summary()
model.fit(mx_train, my_train, epochs=5,verbose=1)

model.evaluate(mx_test, my_test)