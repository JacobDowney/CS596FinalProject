import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def executeT(x_train, y_train, x_test, y_test):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    # Hyperparameters
    epochs = 10000
    learning_rate = 0.01
    n_input = 5
    n_hidden = 10
    n_output = 1
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)
    # Weights
    W1 = tf.Variable(tf.random_uniform([n_input, n_hidden], -1.0, 1.0))
    W2 = tf.Variable(tf.random_uniform([n_hidden, n_output], -1.0, 1.0))
    # Bias
    b1 = tf.Variable(tf.zeros([n_hidden]), name="Bias1")
    b2 = tf.Variable(tf.zeros([n_output]), name="Bias2")
    # Activation functions
    L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
    hy = tf.sigmoid(tf.matmul(L2, W2) + b2)
    # Optimizer
    #cost = tf.reduce_mean(-Y*tf.log(hy) - (1-Y)*tf.log(1-hy))
    cost = tf.reduce_mean((Y - hy) * (Y - hy))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    # Initializer
    init = tf.initialize_all_variables()
    with tf.Session() as session:
        session.run(init)
        for step in range(0, epochs):
            session.run(optimizer, feed_dict={X: x_train, Y: y_train})
            if step % 1000 == 0:
                print (session.run(cost, feed_dict={X: x_train, Y: y_train}))
        answer = tf.equal(tf.floor((hy * 10) + 0.5), tf.floor((Y * 10) + 0.5))
        accuracy = tf.reduce_mean(tf.cast(answer, "float"))
        # Running session
        print(session.run([hy], feed_dict={X: x_train, Y: y_train}))
        print("Accuracy:", accuracy.eval({X: x_train, Y: y_train}))


def execute(x_train, y_train, x_test, y_test):
    # Hidden features
    hidden_neural_units = [10, 5]
    activation_functions = ['relu', 'sigmoid']

    # Parameters
    learning_rate = 0.01
    training_iterations = 100

    # Network Parameters
    n_input = 5
    n_output = 1
    output_function = tf.nn.sigmoid

    # Tensorflow keras Sequential model
    model = tf.keras.Sequential()

    # 2D image we flatten to 1D image to give it to 1 row of neurons
    model.add(tf.keras.Input(shape=(n_input,)))
    # For each amount of neurons in each layer, add it to the model
    for units, act_func in zip(hidden_neural_units, activation_functions):
        # Dense makes neuron in previous row is connected to each neuron this row
        model.add(tf.keras.layers.Dense(units, activation=act_func))
    # 10 output layers from 0-9
    # activation is softmax to find max of each of those neurons and thats the num
    # Output function = softmax
    model.add(tf.keras.layers.Dense(n_output, activation=output_function))

    model.compile(optimizer = tf.optimizers.Adam(learning_rate=learning_rate),
                    loss = tf.keras.losses.MeanAbsoluteError(),
                    metrics = ['accuracy'])

    # Train the Model
    model.fit(x_train, y_train, epochs=training_iterations)

    # Test the Model
    print("TESTING")
    #print(model.evaluate(x_test, y_test))
    raw_predictions = model.predict(x=x_test)
    # predictions = []
    avg = 0.0
    print(raw_predictions)
    for i in range(0, len(raw_predictions)):
        avg += abs(raw_predictions[0] - y_test[i])
    print("AVERAGE MISS: ", (avg / int(len(y_test))))
    #
    # conf_matrix, accuracy, recall_array, precision_array = func_confusion_matrix(predictions, y_test)
    # print("RESULTS")
    # print("Confusion matrix\n", conf_matrix)
    # print("Accuracy: ", accuracy)
    # print("Recall Array: ", [ float('%.3f' % elem) for elem in list(recall_array) ])
    # print("Precision Array: ", [ float('%.3f' % elem) for elem in list(precision_array) ])
    # print("\n\n")
