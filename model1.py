import numpy as np
import tensorflow as tf

# Parameters
learning_rate = 0.01
training_epochs = 5
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 10
n_hidden_2 = 5
n_input = 20
n_output = 1

# tensorflow Graph input
X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_output])



# Tensorflow keras Sequential model
model = tf.keras.Sequential()
# 2D image we flatten to 1D image to give it to 1 row of neurons
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# For each amount of neurons in each layer, add it to the model
for i in hiddenNeuralUnits:
    # Dense makes neuron in previous row is connected to each neuron this row
    # 1 hidden layer and it has 128 neruons
    # activation function = relu, activates pos values and cancels negative
    model.add(tf.keras.layers.Dense(i, activation=activationFunction))
# 10 output layers from 0-9
# activation is softmax to find max of each of those neurons and thats the num
# Output function = softmax
model.add(tf.keras.layers.Dense(numOutputLayers, activation=outputFunction))

model.compile(optimizer = tf.optimizers.Adam(learning_rate=learningRate),
                loss = 'sparse_categorical_crossentropy',
                metrics = ['accuracy'])

# Train the Model
model.fit(x_train, y_train, epochs=trainingIterations)
