import numpy as np
import pandas as pd
from test.test_asyncio.test_events import data_file
from array import array
data = pd.read_csv('train.csv', sep=',', lineterminator="\n")

cols = 6
rows = len(data)

tablecolrow = []

for j, c in enumerate(data):
    tablecolrow.append([])
    for i, r in enumerate(data[c]):
        tablecolrow[j].append(r)

for i in range(0, rows):
    for j in range(0, 6):
        print(tablecolrow[j][i], end=' ')
    print("", end='\n')
    
    
tablerowcol = nesto(tablecolrow)