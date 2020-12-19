import numpy as np
import tensorflow as tf
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras import Model
from keras.layers import Layer
from keras import backend as K

### AUTHOR: Will Patterson

#building a custom RBF neural network with backend Tensorflow
class RBFLayer(Layer):
   def __init__(self, units, gamma, **kwargs):
       super(RBFLayer, self).__init__(**kwargs)
       self.units = units
       self.gamma = K.cast_to_floatx(gamma)

   def build(self, input_shape):
       self.mu = self.add_weight(name='mu',
                                 shape=(int(input_shape[1]), self.units),
                                 initializer='uniform',
                                 trainable=True)
       super(RBFLayer, self).build(input_shape)

   def call(self, inputs):
       diff = K.expand_dims(inputs) - self.mu
       l2 = K.sum(K.pow(diff, 2), axis=1)
       res = K.exp(-1 * self.gamma * l2)
       return res

   def compute_output_shape(self, input_shape):
       return (input_shape[0], self.units)

# Executing a radial basic neural network
def execute(x_train, y_train, x_test, y_test):
    training_iterations = 25
    # implementation of RBF layer. gamma is changed in second parameter of "RBFLayer()"
    model = Sequential()
    model.add(Dense(5, activation='relu'))
    model.add(RBFLayer(20, 0.5))
    model.add(Dense(1, activation='sigmoid', name='output'))

    model.compile(optimizer='rmsprop', loss='MSE')
    model.fit(x_train, y_train, epochs=training_iterations)

    y_pred = model.predict(x=x_test)
    return [float(y[0]) for y in y_pred]
