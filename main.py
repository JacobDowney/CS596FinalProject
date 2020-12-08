# import statements
from helpers import parseData, normalizeData, convertToNumpy

data2018 = "mlb-player-stats-Batters-2018.csv"
data2019 = "mlb-player-stats-Batters-2019.csv"
data2020 = "mlb-player-stats-Batters-2020.csv"

def main():
    csvFileName = data2018

    fields, players = parseData(csvFileName)
    normalizedData = normalizeData(players)
    playerNames, numpyData = convertToNumpy(normalizedData)
    print(playerNames[0])
    print(numpyData[0])

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
