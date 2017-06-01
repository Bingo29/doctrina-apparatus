print("Ovdje pisite rjesenje za NLP")
print("Lovre, tvoj zadatak je da pripremis podatke za NLP.")
print("Mozes koristiti pripremljene podatke iz Bag of Words --> GIT")

import pandas as pd

data = pd.read_csv('train.csv', sep=',', lineterminator="\n")

cols = 6
rows = len(data)

tablecolrow = []
tablerowcol = []

for j, c in enumerate(data):
    tablecolrow.append([])
    for i, r in enumerate(data[c]):
        tablecolrow[j].append(r)

# This takes less memory than np.transpose
tablerowcol = list(zip(*tablecolrow))

for i in range(0, rows):
    for j in range(0, 6):
        print(tablerowcol[i][j], end=' ')
    print("", end='\n')