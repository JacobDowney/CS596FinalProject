# import statements
from helpers import getParsedNormalizedData, splitData
import model1
import numpy as np

data2018 = "mlb-player-stats-Batters-2018.csv"
data2019 = "mlb-player-stats-Batters-2019.csv"
data2020 = "mlb-player-stats-Batters-2020.csv"

def main():
    csvFileName = data2018
    percentTraining = 0.8

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

    #trainAndTest(x_train, y_train, x_test, y_test)

    model1.execute(x_train, y_train, x_test, y_test)

    # MODEL # 1 JACOB ->
    # Jacobs models functions
    # Printing models outputs

    # MODEL # 2 WILL ->
    # Wills models functions
    # Printing models outputs

    # MODEL # 3 MATT ->
    # Matts models functions
    # Printing models outputs


    print("done")


if __name__ == '__main__':
    main()
else:
    print("Service not provided")
