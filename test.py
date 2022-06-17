from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import tensorflow as tf
import numpy as np

inputs = np.loadtxt("inputs.txt")
print(inputs.shape)

relevant_dimensions= len(inputs[0])
# print(inputs[0])
model = tf.keras.models.load_model('my_model.h5')
# model.summary()

print(model.predict(inputs[0]))