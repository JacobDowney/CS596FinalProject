# import statements
from helpers import getParsedNormalizedData, splitData, scorePredictions
import model1
import model2
import model3
import numpy as np
import pickle

data2018 = "mlb-player-stats-Batters-2018.csv"
data2019 = "mlb-player-stats-Batters-2019.csv"
data2020 = "mlb-player-stats-Batters-2020.csv"


def main():
    csvFileName = data2018
    percentTraining = 0.8

    fields, playerNames, numpyData, playerScores = getParsedNormalizedData(csvFileName)
    numTrain = int(percentTraining * len(numpyData))
    x_train, y_train, x_test, y_test = splitData(numpyData, playerScores, numTrain)

    print('Fields used for predicting score:', fields)
    print("Starting model executions\n")

    # MODEL # 1 JACOB -> Feedforward Nerual Network
    predModel1 = model1.execute(x_train, y_train, x_test, y_test)
    median, mean, std_dev, indexOf95 = scorePredictions(predModel1, y_test)
    printMetrics('Feedforward Neural Network', median, mean, std_dev, indexOf95)

    # MODEL # 2 WILL -> Radial Basic Model
    predModel2 = model2.execute(x_train, y_train, x_test, y_test)
    median, mean, std_dev, indexOf95 = scorePredictions(predModel2, y_test)
    printMetrics('Radial Basic Neural Network', median, mean, std_dev, indexOf95)

    # MODEL # 3 MATT -> Support Vector Machine
    predModel3 = model3.execute(x_train, y_train, x_test, y_test, fields)
    median, mean, std_dev, indexOf95 = scorePredictions(predModel3, y_test)
    printMetrics('Support Vector Machine', median, mean, std_dev, indexOf95)

    print("done")

def printMetrics(name, median, mean, std_dev, indexOf95):
    print('Model:', name)
    print('Median Absolute Error:', median)
    print('Mean Absolute Error:', mean)
    print('Standard Deviation:', std_dev)
    prnt = 'We can say with 95% certainty that prediction x is within x -'
    print(prnt, indexOf95, 'and x +', indexOf95)


if __name__ == '__main__':
    main()
else:
    print("Service not provided")
