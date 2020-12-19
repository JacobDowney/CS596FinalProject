import csv
import os
import random
import math
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

"""
Model scoring functions
"""

# Used for calculating confusion matrix and precision and accuracy for model
def scorePredictions(predictions, answers, percentError):
    assert len(predictions) == len(answers), "Num predictions must equal num answers"
    import matplotlib.pyplot as plt

    # This is the only variable we should be changing if the model is more or
    # less accurate
    gaps = 0.005

    # Code for getting values for chart
    groupings = {}
    i = 0.0
    diffs = [abs(predictions[i] - answers[i]) for i in range(0, len(answers))]
    diffs.sort()

    median = diffs[int(len(diffs) / 2)]
    if len(diffs) % 2 != 0:
        median = round((diffs[int(len(diffs)/2)] + diffs[int((len(diffs)/2))+1]) / 2.0, 3)
    print('Median Absolute Error:', median)
    mean = sum(diffs) / len(diffs)
    print('Mean Absolute Error:', sum(diffs) / len(diffs))
    std_dev = sum([(x*10000) * (x*10000) for x in diffs])
    std_dev = math.sqrt((1.0 / len(diffs)) * std_dev) / 10000.0
    print('Standard Deviation:', std_dev)
    indexOf95 = int(0.95 * len(diffs))
    prnt = 'We can say with 95% certainty that prediction x is within x -'
    print(prnt, round(diffs[indexOf95], 4), 'and x +', round(diffs[indexOf95], 4))

    max_diffs = diffs[-1] + gaps
    while i < max_diffs:
        groupings[round(i, 3)] = 0
        i += gaps

    for i in range(0, len(answers)):
        unit = round(math.floor(diffs[i] * (1 / gaps)) / float(1 / gaps), 3)
        groupings[unit] += 1

    xplot = []
    yplot = []
    for key, value in groupings.items():
        xplot.append('(' + str(key) + ',' + str(round(key+gaps, 3)) + ')')
        yplot.append(value)

    plt.barh(xplot, yplot, align='center') #, alpha=0.001)
    plt.xticks(np.arange(0, max(yplot)+1, math.ceil(len(yplot) / 10)))
    plt.yticks(np.arange(0, len(yplot), 2))
    plt.xlabel('Number of Predictions in Range')
    plt.ylabel('Range of Mean Abolsute Error')
    plt.title('Range of Accuracy of Predictions by Model')
    plt.show()
