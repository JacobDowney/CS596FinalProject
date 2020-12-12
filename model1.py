import numpy as np
import tensorflow as tf

def execute(x_train, y_train, x_test, y_test):
    # Hidden features
    hidden_neural_units = [10]
    activation_functions = ['relu']

    # Parameters
    learning_rate = 0.01
    training_iterations = 10

    # Network Parameters
    n_input = 5
    n_output = 1
    output_function = tf.nn.sigmoid

    # Tensorflow keras Sequential model
    model = tf.keras.Sequential()

    # 2D image we flatten to 1D image to give it to 1 row of neurons
    model.add(tf.keras.Input(n_input,None))
    # For each amount of neurons in each layer, add it to the model
    for unit, act_func in zip(hidden_neural_units, activation_functions):
        # Dense makes neuron in previous row is connected to each neuron this row
        model.add(tf.keras.layers.Dense(unit, activation=act_func))
    # 10 output layers from 0-9
    # activation is softmax to find max of each of those neurons and thats the num
    # Output function = softmax
    model.add(tf.keras.layers.Dense(n_output, activation=output_function))

    model.compile(optimizer = tf.optimizers.Adam(learning_rate=learning_rate),
                    loss = 'sparse_categorical_crossentropy',
                    metrics = ['accuracy'])

    # Train the Model
    model.fit(x_train, y_train, epochs=training_iterations)

    # Test the Model
    print("TESTING")
    #print(model.evaluate(x_test, y_test))
    raw_predictions = model.predict(x=x_test)
    # predictions = []
    for i in range(0, len(raw_predictions)):
        print(raw_predictions[i][0], y_test[i])
    #
    # conf_matrix, accuracy, recall_array, precision_array = func_confusion_matrix(predictions, y_test)
    # print("RESULTS")
    # print("Confusion matrix\n", conf_matrix)
    # print("Accuracy: ", accuracy)
    # print("Recall Array: ", [ float('%.3f' % elem) for elem in list(recall_array) ])
    # print("Precision Array: ", [ float('%.3f' % elem) for elem in list(precision_array) ])
    # print("\n\n")
