import numpy as np
import tensorflow as tf

f = lambda x: 2*x
Xtrain = np.random.rand(400,1)
ytrain = f(Xtrain)
Xval = np.random.rand(200,1)
yval = f(Xval)

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='relu')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError()
             )

model.fit(Xtrain, ytrain, epochs=50, verbose=1, validation_data=(Xval, yval))

prediction = model.predict(x=np.array([2,4,6,2,3,5]))

print(prediction)
