import matplotlib.pyplot as plt
import sklearn.svm as svm

### AUTHOR: Matthew Orgon

# Execute function for svm model.
# Note: This took a lot more code to make but after editing it was removed for
# this clean final version
def execute(x_train, y_train, x_test, y_test, fields, showGraph=0):
    # kernel_types = ['linear', 'poly', 'rbf'] #, 'sigmoid']
    clf = svm.SVR(kernel='linear', tol=.0005, epsilon=.0005)
    y_pred = clf.fit(x_train, y_train).predict(x_test)

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

    return [float(y) for y in y_pred]
