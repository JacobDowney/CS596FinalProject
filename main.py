# import statements
from helpers import getParsedNormalizedData, splitData, scorePredictions
import model1
import model2
import model3
import time

### AUTHOR: All of us. Jacob, Matt, and Will

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
    start1 = time.time()
    predModel1 = model1.execute(x_train, y_train, x_test, y_test)
    median1, mean1, std_dev1, indexOf951 = scorePredictions(predModel1, y_test)
    end1 = time.time()

    # MODEL # 2 WILL -> Radial Basic Model
    start2 = time.time()
    predModel2 = model2.execute(x_train, y_train, x_test, y_test)
    median2, mean2, std_dev2, indexOf952 = scorePredictions(predModel2, y_test)
    end2 = time.time()


    # MODEL # 3 MATT -> Support Vector Machine
    start3 = time.time()
    predModel3 = model3.execute(x_train, y_train, x_test, y_test, fields)
    median3, mean3, std_dev3, indexOf953 = scorePredictions(predModel3, y_test)
    end3 = time.time()

    printMetrics('Feedforward Neural Network', median1, mean1, std_dev1, indexOf951, end1-start1)
    printMetrics('Radial Basic Neural Network', median2, mean2, std_dev2, indexOf952, end2-start2)
    printMetrics('Support Vector Machine', median3, mean3, std_dev3, indexOf953, end3-start3)
    print("done")

def printMetrics(name, median, mean, std_dev, indexOf95, time):
    print('Model:', name)
    print('Median Absolute Error:', median)
    print('Mean Absolute Error:', mean)
    print('Standard Deviation:', std_dev)
    print('Train and Predict Time', time)
    prnt = 'We can say with 95% certainty that prediction x is within x -'
    print(prnt, indexOf95, 'and x +', indexOf95)


if __name__ == '__main__':
    main()
else:
    print("Service not provided")
