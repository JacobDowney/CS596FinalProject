import csv
import os
import random
import time
import numpy as np

def splitData(x, y, numTrain):
    rng_state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(rng_state)
    np.random.shuffle(y)
    return x[:numTrain], y[:numTrain], x[numTrain:], y[numTrain:]

def getParsedNormalizedData(fileName):
    fields, players, allData, realData = parseData(fileName)
    return fields, players, np.array(normalizeData(realData)), np.array(scorePlayers(allData))

# Takes in a file name and returns a dictionary of player names to list of stats
def parseData(fileName):
    if os.path.exists(fileName):
        try:
            f = open(fileName, 'r')
        except:
            print("Could not open/read file", fileName)
            sys.exit()
        with f:
            reader = csv.reader(f)
            next(reader) # First line is just header information
            players = []
            data = []
            realData = []
            for row in reader:
                players.append(row[0])
                rowData = ([float(i) for i in row[3:]])
                data.append(rowData)
                realData.append([rowData[1], rowData[2], rowData[16], rowData[17], rowData[18]])
            return ['G', 'AB', 'AVG', 'OBP', 'SLG'], players, data, realData
    else:
        return [], {}, [], []

def scorePlayers(data):
    score = lambda d : d[4] + (2*d[5]) + (3*d[6]) + (4*d[7]) + d[3] + d[8] + d[11] + d[15] + (2*d[9]) - d[10]
    scores = [score(d) for d in data]
    maxScore = max(scores) + 1
    return [x / maxScore for x in scores]

# Takes dictionary of (playerName -> list of stats) and finds max and mins of
# each stat for normalization purposes and returns a dictionary of
# (playerName -> list of normalized stats ranged (0, 1.0))
def normalizeData(data):
    # Gettin mins and maxs
    mins = list(data)[0][:]
    maxs = list(data)[0][:]
    for values in data:
        for i in range(0, len(values)):
            mins[i] = min(mins[i], values[i])
            maxs[i] = max(maxs[i], values[i])

    dataNormalized = []
    for vals in data:
        normalized = [((vals[i]-mins[i]) / (maxs[i]-mins[i]+1)) for i in range(0, len(vals))]
        dataNormalized.append(normalized)
    return dataNormalized

##### Problem 3: Comparative Studies
# Please write a function to calculate the confusion matrix for the prediction
# results of a classifier. The function should take the form:
# def func_calConfusionMatrix(predY, trueY) where predY is the vector of the
# predicted labels and trueY is the vector of true labels. This function should
# return accuracy, per-class precision, and per-class recall rate.
# NOTE: these are meant for normal pyhton lists not np.arrays
def accuracies(predY, trueY):
    if len(predY) != len(trueY):
        print('length of predY and trueY must be equal')
        return -1, [], []

    # Calcualte accuracy. Basically (TP+TN) / TOTAL
    accuracy025 = 0
    accuracy05 = 0
    accuracy10 = 0
    accuracy25 = 0
    for i in range(0, len(trueY)):
        diff = abs(predY[i] - trueY[i])
        if diff < 0.025:
            accuracy025 += 1
        if diff < 0.05:
            accuracy05 += 1
        if diff < 0.1:
            accuracy10 += 1
        if diff < 0.25:
            accuracy25 += 1
    accuracy025 = float(accuracy025) / float(len(trueY))
    accuracy05 = float(accuracy05) / float(len(trueY))
    accuracy10 = float(accuracy10) / float(len(trueY))
    accuracy25 = float(accuracy25) / float(len(trueY))

    return [accuracy025, accuracy05, accuracy10, accuracy25]

    # # for each type compute precision and recall rates
    # precision = []
    # recall = []
    # for i in range(0, len(types)):
    #     type = types[i]
    #     true_positive = 0
    #     false_positive = 0
    #     false_negative = 0
    #     for i in range(0, len(trueY)):
    #         if type == trueY[i] and predY[i] == trueY[i]:
    #             true_positive += 1
    #         if type == trueY[i] and predY[i] != trueY[i]:
    #             false_negative += 1
    #         if type == predY[i] and predY[i] != trueY[i]:
    #             false_positive += 1
    #     precision.append(float(true_positive) / float(true_positive + false_positive))
    #     recall.append(float(true_positive) / float(true_positive + false_negative))
    #
    # return accuracy, precision, recall
