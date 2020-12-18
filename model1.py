import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from helpers import accuracies

def testingParameters():
    # 15, 10, 5 combinations
    h = [[15, 15], [15, 10], [15, 5], [10, 10], [10, 5], [5, 5]]
    # relu, sigmoid, softmax, tanh, exponential
    a = ['relu', 'sigmoid', 'exponential']

    averages = []
    for hidden in h:
        for act1 in a:
            print("\n\n\n\n\nSTARTING: ", hidden, act1)
            for act2 in a:
                for act3 in a:
                    avg = 0.0
                    for i in range(0, 5):
                        avg += execute(x_train,y_train,x_test,y_test,hidden,[act1,act2],act3)
                    avg = avg / 5.0
                    averages.append([avg, hidden, act1, act2, act3])
    pickle.dump(averages, open('avg.pkl', 'wb'))


def sortResults():
    avg = pickle.load( open('avg.pkl', 'rb') )
    sort = sorted(avg, key=lambda x: x[0])
    for a in sort:
        print(a)


def execute(x_train, y_train, x_test, y_test):
    # Hidden features
    hidden_neural_units = [15, 15]
    activation_functions = ['relu', 'relu']

    # Parameters
    learning_rate = 0.01
    training_iterations = 50

    # Network Parameters
    n_input = 5
    n_output = 1
    output_function = 'exponential'

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
    raw_predictions = model.predict(x=x_test)
    predictions = []
    avg = 0.0
    for i in range(0, len(raw_predictions)):
        predictions.append(raw_predictions[i][0])
        avg += abs(raw_predictions[i][0] - y_test[i])
    avg = avg / int(len(y_test))
    print(avg)

    acc = accuracies(predictions, y_test)
    print(acc)

    # conf_matrix, accuracy, recall_array, precision_array = func_confusion_matrix(predictions, y_test)
    # print("RESULTS")
    # print("Confusion matrix\n", conf_matrix)
    # print("Accuracy: ", accuracy)
    # print("Recall Array: ", [ float('%.3f' % elem) for elem in list(recall_array) ])
    # print("Precision Array: ", [ float('%.3f' % elem) for elem in list(precision_array) ])
    # print("\n\n")
