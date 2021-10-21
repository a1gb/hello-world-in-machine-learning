import tensorflow as tf
import numpy as np
from tensorflow import keras

# Define and compile the neural network
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

# Provide the data (where y = 2x + 1)
x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
y = np.array([-1.0, 1.0, 3.0, 5.0, 7.0, 9.0], dtype=float)

# Train the neural network
model.fit(x, y, epochs=500)

# Use the model to make predictions
print(model.predict([10.0]))
