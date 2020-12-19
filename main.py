# import statements
from helpers import getParsedNormalizedData, splitData, scorePredictions
import model1
import model3
import numpy as np
import pickle

data2018 = "mlb-player-stats-Batters-2018.csv"
data2019 = "mlb-player-stats-Batters-2019.csv"
data2020 = "mlb-player-stats-Batters-2020.csv"


def main():
    csvFileName = data2018
    percentTraining = 0.8

    # Percent error that is acceptable for a correct prediciton. For example,
    # If the correct answer is 0.55, with a prediction error of 0.5 a prediction
    # from 0.5 to 0.6 would be correct, else incorrect.
    # Correct_Answer = abs(pred_y - y_test) < predictionError
    predictionError = 0.05

    fields, playerNames, numpyData, playerScores = getParsedNormalizedData(csvFileName)
    numTrain = int(percentTraining * len(numpyData))
    x_train, y_train, x_test, y_test = splitData(numpyData, playerScores, numTrain)

    print(fields)
    print(playerNames[0]) # Not that first name doesn't align with trian data

    print(x_train[0])
    print(y_train[:5])
    print(len(x_test))
    print(len(y_test))
    print("Starting model executions\n")

    # MODEL # 1 JACOB ->
    # Jacobs models functions
    # Printing models outputs
    # predModel1 = model1.execute(x_train, y_train, x_test, y_test)
    # scorePredictions(predModel1, y_test, predictionError)

    # MODEL # 2 WILL ->
    # Wills models functions
    # Printing models outputs

    # MODEL # 3 MATT ->
    # Matts models functions
    # Printing models outputs

    #model3.execute(x_train, y_train, x_test, y_test)

    predModel3 = model3.execute(x_train, y_train, x_test, y_test, fields)
    scorePredictions(predModel3, y_test, predictionError)

    # model3.execute(x_train, y_train, x_test, y_test, fields)




    print("done")


if __name__ == '__main__':
    main()
else:
    print("Service not provided")
