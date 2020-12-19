import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
import sklearn.svm as svm
from sklearn.svm import SVR
import numpy as np
import sklearn.metrics as metrics

from sklearn import utils


def execute(x_train, y_train, x_test, y_test, fields, showGraph=0):
    # kernel_types = ['linear', 'poly', 'rbf'] #, 'sigmoid']
    clf = svm.SVR(kernel='linear', tol=.0005, epsilon=.0005)
    y_pred = clf.fit(x_train, y_train).predict(x_test)

    # mean_absolute_error = metrics.mean_absolute_error(y_test, y_pred)
    # print("mean absolute error:", mean_absolute_error)

    if showGraph:
        plt.scatter(y_test, y_pred, color='darkorange', label='data')
        x=[0,1]
        y = x
        plt.plot(x, y)
        plt.xlabel('data')
        plt.ylabel('target')
        plt.title('Support Vector Regression')
        plt.legend()
        plt.show()

    return y_pred
