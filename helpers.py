import csv
import os

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
            fields = []
            players = {}
            i = 0
            for row in reader:
                if i == 0:
                    i = 1
                    fields = row[3:]
                else:
                    players[row[0]] = [float(i) for i in row[3:]]
            return fields, players
    else:
        return [], {}

# Takes dictionary of (playerName -> list of stats) and finds max and mins of
# each stat for normalization purposes and returns a dictionary of
# (playerName -> list of normalized stats ranged (0, 1.0))
def normalizeData(players):
    playersNormalized = {}
    maxmins = []
    for pValue in list(players.values())[0]:
        maxmins.append([pValue, pValue])

    playerNames = players.keys()
    for playerName in playerNames:
        values = players[playerName]
        for i in range(0, len(values)):
            maxmins[i][0] = min(maxmins[i][0], values[i])
            maxmins[i][1] = max(maxmins[i][1], values[i])
    for playerName in playerNames:
        values = players[playerName]
        newValues = []
        for i in range(0, len(values)):
            newValues.append((values[i] - maxmins[i][0]) / (maxmins[i][1] - maxmins[i][0]))
        playersNormalized[playerName] = newValues
    return playersNormalized
