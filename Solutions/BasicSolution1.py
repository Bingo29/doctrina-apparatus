import pandas as pd
import re
import string

from nltk.corpus import stopwords # Import the stop word list
cachedStopWords = set(stopwords.words("english"))
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

#remove punctuation marks from question
#returns lower case string
def RemovePunctuationCharacters(strQuestion):
    strRetval = str(strQuestion)
    for char in string.punctuation:
        strRetval = strRetval.replace(char, "")
        
    strRetval = re.sub(' +', ' ', strRetval)
    return strRetval.lower()

#removes unneccecery parts of questions in order to prepare it for bag of words
#returns filtered table
def FilterTable(tbTableRowCol):
    tbRetval = tbTableRowCol
    for i, row in enumerate(tbRetval):
        tbRetval[i] = list(row)
        
    for i in range(0, len(tbTableRowCol)):
        for j in range(0, len(tbTableRowCol[0])):
            if j == 3 or j == 4:
                punctuationlessQuestion = RemovePunctuationCharacters(tbTableRowCol[i][j]) 
                tbRetval[i][j] = ' '.join([word for word in punctuationlessQuestion.split() if word not in cachedStopWords])
                tbRetval[i][j] =" ".join([lemmatizer.lemmatize(i) for i in tbRetval[i][j].split()])
    return tbRetval

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

tablerowcol = FilterTable(tablerowcol)

for i in range(0, rows):
    for j in range(0, 6):
        print(tablerowcol[i][j], end=' ')
    print("", end='\n')

    



