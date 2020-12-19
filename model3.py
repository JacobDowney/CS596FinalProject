import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
import sklearn.svm as svm
from sklearn.svm import SVR
import numpy as np
import sklearn.metrics as metrics

from sklearn import utils


def execute(x_train, y_train, x_test, y_test, fields):
    final_x = x_test[:]
    lab_enc = preprocessing.LabelEncoder()
    #
    # encodedtrain = lab_enc.fit_transform(y_train)
    # encodedtest = lab_enc.fit_transform(y_test)
    num_iterations = 5
    curr_iters = 0
    kernel_types = ['linear', 'poly', 'rbf']#, 'sigmoid']
    svm_c_error = []
    sum_absolute_error = 0
    avg_err =0
    #bestModelParameterTest
    clf = svm.SVR(kernel='linear')#kernel='linear', tol=.0005, epsilon=.0005)
    y_pred = clf.fit(x_train, y_train).predict(x_test)
    mean_absolute_error = metrics.mean_absolute_error(y_test, y_pred)
    print("mean absolute error:", mean_absolute_error)
    plt.scatter(y_test, y_pred, color='darkorange', label='data')
    x=[0,1]
    y = x
    plt.plot(x, y)
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
    # print(, ':', mean_absolute_error)
    # mean_errors.append(mean_absolute_error)

    # test_epsilon = [.00001,.0001,.0005,.001,.005,.01]
    # test_tolerance = [.0000001, .000001,.00005,.0001,.0005,.001,.005]
    # mean_errors = []
    # for t in test_tolerance:
    #     clf = svm.SVR(kernel='linear', tol=t)
    #     y_pred = clf.fit(x_train, y_train).predict(x_test)
    #     mean_absolute_error = metrics.mean_absolute_error(y_test, y_pred)
    #     print(t,':', mean_absolute_error)
    #     mean_errors.append(mean_absolute_error)
    #
    # plt.plot(test_tolerance, mean_errors)
    # plt.title('SVM by epsilons')
    # plt.xlabel('epsilon')
    # plt.ylabel('error')
    # plt.xticks(test_tolerance)
    # plt.show()

    # for k in kernel_types:
    #     clf = svm.SVR(kernel=k)
    #     sum_absolute_error = 0
    #     curr_iters = 0
    #     while(curr_iters < num_iterations):
    #
    #         if (k == 'linear'):
    #             y_pred = clf.fit(x_train, y_train).predict(x_test)
    #         if (k == 'poly'):
    #             y_pred = clf.fit(x_train, y_train).predict(x_test)
    #         if (k == 'rbf'):
    #             y_pred = clf.fit(x_train, y_train).predict(x_test)
    #         confidence = clf.score(x_test, y_test)
    #         # y_pred = clf.predict(x_test)
    # #        encodedy_predict = lab_enc.fit_transform(y_pred)
    # #        print(y_pred)
    #         mean_absolute_error = metrics.mean_absolute_error(y_test, y_pred)
    #         sum_absolute_error += mean_absolute_error
    #
    #
    #         mse = metrics.mean_squared_error(y_test, y_pred)
    #         r2 = metrics.r2_score(y_test, y_pred)
    #         explained_variance = metrics.explained_variance_score(y_test, y_pred)
    #
    #         svm_c_error.append(1-confidence)
    #
    #         # c_range = 10, 50, 100, 500, 1000  #
    #         # svm_c_error = []
    #         # for c_value in c_range:
    #         #     model = svm.SVR(kernel= k, C=c_value)
    #         #     model.fit(X=x_train, y=encodedtrain)
    #         #     mean_absolute_errorC = metrics.mean_absolute_error(y_test, y_pred)
    #         #     svm_c_error.append(mean_absolute_errorC)
    #         # print(svm_c_error)
    #         # plt.plot(c_range, svm_c_error)
    #         # plt.title(k)
    #         # plt.xlabel('c values')
    #         # plt.ylabel('error')
    #         # plt.xticks(c_range)
    #         # plt.show()
    #
    #         # kernel_types = ['linear', 'poly', 'rbf']
    #         # svm_kernel_error = []
    #         # for kernel_value in kernel_types:
    #         #     # your own codes
    #         #     model2 = svm.SVC(kernel=kernel_value, C=6)
    #         #     model2.fit(X=x_train, y=y_train)
    #         #     error = 1. - model2.score(x_validation, y_validation)
    #         #     svm_kernel_error.append(error)
    #
    #
    #
    #
    #
    #
    #             # error = 1. - model.score(x_test, encodedtest)
    #             # svm_c_error.append(error)
    #
    #         print(k, "mean absolute error: ", mean_absolute_error)
    #         # print(k, round(confidence, 4))
    #         # print("explained variance : ", round(explained_variance, 4))
    #         # print("mean absolute error: ", round(mean_absolute_error, 4))
    #         # print("mean squre error   : ", round(mse, 4))
    #         # print("r2                 : ", round(r2, 4))
    #         print()
    #         curr_iters += 1
    #     avg_err = sum_absolute_error / num_iterations
    #     print("avg error :" , avg_err)



  #   plt.plot(kernel_types, svm_c_error)
  #   plt.title('SVM by Kernels')
  #   plt.xlabel('Kernel')
  #   plt.ylabel('error')
  #   plt.xticks(kernel_types)
  #   plt.show()
  #
  #   x_axis = []
  #   np.array(x_axis)
  # #  x_test.transpose()
  #   i =0
  #   colors = ['red', 'blue', 'green', 'orange', 'pink']
  #   new_arr = x_test.transpose()
  #   for r in new_arr:
  #       plt.scatter(r, y_test, color=colors[i], label=fields[i])
  #       i+= 1
  #
  #
  #
  #   lw = .5
  #   x_axis = []
  #
  #
  #  # plt.scatter(x_test, y_test, color='darkorange', label='data')
  #  #  plt.hold('on')
  #   plt.plot(y_test, y_rbf, color='navy', lw=lw, label='RBF model')
  #  #  plt.plot(y_test, y_linear, color='c', lw=lw, label='Linear model')
  #  #  plt.plot(new_arr[0], y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
  #   plt.xlabel('data')
  #   plt.ylabel('target')
  #   plt.title('Support Vector Regression')
  #   plt.legend()
  #   plt.show()



        # print(y_test.shape)
        # print(y_pred.shape)
        # print(confusion_matrix(encodedtest, encodedy_predict))
        # print(classification_report(encodedtest, encodedy_predict))














    # print(utils.multiclass.type_of_target(encodedtrain))
    #
    #
    # svclassifier = SVC(kernel='linear')
    # svclassifier.fit(x_train, encodedtrain)
    #
    # y_pred = svclassifier.predict(x_test)
    #
    # print(utils.multiclass.type_of_target(y_pred))
    # print(y_pred)
    #
    # print(utils.multiclass.type_of_target(encodedtest))
    # print(encodedtest)
    #
    # print(confusion_matrix(encodedtest, y_pred))
    # print(classification_report(encodedtest, y_pred))


    #
    # c_range = 2, 4, 6, 8, 10  #
    # svm_c_error = []
    # for c_value in c_range:
    #     model = svm.SVC(kernel='rbf', C=c_value)
    #     model.fit(X=x_train, y=encodedtrain)
    #     error = 1. - model.score(x_test, encodedtest)
    #     svm_c_error.append(error)
    # plt.plot(c_range, svm_c_error)
    # plt.title('poly SVM')
    # plt.xlabel('c values')
    # plt.ylabel('error')
    # plt.xticks(c_range)
    # plt.show()
    #
    # y_pred = model.predict(x_test)
    # print(confusion_matrix(encodedtest, y_pred))
    # print(classification_report(encodedtest, y_pred))

    # # Fit regression model
    # svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    # svr_lin = SVR(kernel='linear', C=100, gamma='auto')
    # svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
    #                coef0=1)
    #
    # # #############################################################################
    # # Look at the results
    # lw = 2
    #
    # svrs = [svr_rbf, svr_lin, svr_poly]
    # kernel_label = ['RBF', 'Linear', 'Polynomial']
    # model_color = ['m', 'c', 'g']
    #
    # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
    # for ix, svr in enumerate(svrs):
    #     axes[ix].plot(x_train, svr.fit(x_train, encodedtrain).predict(x_test), color=model_color[ix], lw=lw,
    #                   label='{} model'.format(kernel_label[ix]))
    #     axes[ix].scatter(x_train[svr.support_], encodedtrain[svr.support_], facecolor="none",
    #                      edgecolor=model_color[ix], s=50,
    #                      label='{} support vectors'.format(kernel_label[ix]))
    #     axes[ix].scatter(x_train[np.setdiff1d(np.arange(len(x_train)), svr.support_)],
    #                      encodedtrain[np.setdiff1d(np.arange(len(x_train)), svr.support_)],
    #                      facecolor="none", edgecolor="k", s=50,
    #                      label='other training data')
    #     axes[ix].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
    #                     ncol=1, fancybox=True, shadow=True)
    #
    # fig.text(0.5, 0.04, 'data', ha='center', va='center')
    # fig.text(0.06, 0.5, 'target', ha='center', va='center', rotation='vertical')
    # fig.suptitle("Support Vector Regression", fontsize=14)
    # plt.show()
