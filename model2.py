import pandas as pd
import numpy as np

import tensorflow as tf
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.losses import binary_crossentropy
from keras import Model

#import matplotlib.pyplot as plt

from keras.layers import Layer
from keras import backend as K

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

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# choosing our 5 parameters, dropping remaining 13 parameters. Our selected parameters are Games, AB, AVG, OBP, SLG
data = pd.read_csv("mlb-player-stats-Batters-2018.csv")
data.drop(columns=["Team","Player","Pos","Age","R","H","2B","3B","HR","RBI","SB","CS","BB","SO","SH","SF","HBP"],inplace=True)
data.head(10)

# normalizing data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(data)
data = pd.DataFrame(scaler.fit_transform(data),columns=data.columns)
data.head(10)


X,y = data.drop(columns="OPS"), data["OPS"]
X = X.to_numpy()
y = y.to_numpy()

target = data.pop('OPS')

dataset = tf.data.Dataset.from_tensor_slices((data.values,target.values))
train_dataset = dataset.shuffle(len(data)).batch(1)

# implementation of RBF layer. gamma is changed in second parameter of "RBFLayer()"
model = Sequential()
model.add(Dense(5, activation='relu'))
model.add(RBFLayer(10, 0.5))
model.add(Dense(1, activation='sigmoid', name='output'))

model.compile(optimizer='rmsprop',loss='MSE')

model.fit(X,y,batch_size=5,epochs=10)

y_pred = model.predict(X)
score = model.evaluate(X,y,verbose=1)

import matplotlib.pyplot as plt
plt.scatter(y_pred, y)

print(score)
