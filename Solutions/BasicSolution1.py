import numpy as np
import pandas as pd
from array import array

def okreniTablicu(tablicacolrow):
    col = len(tablicacolrow)
    row = len(tablicacolrow[0])
    novaTablica = [[None] * col for x in range(0, row)]
    for i in range(0, col):
        for j in range(0, row):
            novaTablica[j][i] = tablicacolrow[i][j]        
return novaTablica;


data = pd.read_csv('train.csv', sep=',', lineterminator="\n")

cols = 6
rows = len(data)

tablecolrow = []
tablerowcol = []

for j, c in enumerate(data):
    tablecolrow.append([])
    for i, r in enumerate(data[c]):
        tablecolrow[j].append(r)

print(tablecolrow[4][5])

tablerowcol = okreniTablicu(tablecolrow)

for i in range(0, rows):
    for j in range(0, 6):
        print(tablerowcol[i][j], end=' ')
    print("", end='\n')    

print(tablerowcol[4][5])