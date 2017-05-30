import pandas as pd
import re
import string
from sklearn import svm
from sklearn.externals import joblib
from timeit import default_timer as timer
from scipy.sparse import hstack, vstack

from nltk.corpus import stopwords # Import the stop word list
from numpy import array
from sklearn.svm.classes import SVC
cachedStopWords = set(stopwords.words("english"))
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix

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
        if i % 1000 == 0:
            print(i)
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

tablecolrow = None

tablerowcol = FilterTable(tablerowcol)
'''
for i in range(0, rows):
    for j in range(0, 6):
        print(tablerowcol[i][j], end=' ')
    print("", end='\n')
'''

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 20000) 

    

q1 = []
q2 = []

for row in tablerowcol:
    q1.append(row[3])
    q2.append(row[4])
    
questionBase = [q1, q2]

allQuestions = q1 + q2

vectorizer.fit(allQuestions)

znacajkePitanja = [vectorizer.transform(q1), vectorizer.transform(q2)]

print("Bag of words done")

for i in range(0, znacajkePitanja[0].shape[0]):
    tablerowcol[i][3] = znacajkePitanja[0][i]
    tablerowcol[i][4] = znacajkePitanja[1][i]
    if i % 1000 == 0:
        print(i)

znacajkePitanja = None

print("Gotovo")

#learning data = X, Y
svmLearningData = [[], []]

for i, row in enumerate(tablerowcol):
    svmLearningData[0].append(hstack([row[3], row[4]]).tocsr())
    svmLearningData[1].append(int(row[5]))
    if i % 1000 == 0:
        print(i)

svmKlasifikator = SVC(kernel='rbf', verbose=True, probability=True, max_iter=1000000)
print("Sparse")
Xsparse = vstack(svmLearningData[0]).tocsr()
print("Learning started")
tmStart = timer()
svmKlasifikator.fit(Xsparse, svmLearningData[1])
tmEnd = timer()
print("Learning ended")
print("Learning lasted", tmEnd - tmStart)

joblib.dump(svmKlasifikator, 'BagOfWordsSVMNauceni.pkl') 

print("Spremljen je napredak ucenja")

svmUcitani = joblib.load('BagOfWordsSVMNauceni.pkl') 