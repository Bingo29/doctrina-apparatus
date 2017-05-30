import re
import string
from timeit import default_timer as timer

from nltk.corpus import stopwords  # Import the stop word list
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm.classes import SVC

import pandas as pd
import os


cachedStopWords = set(stopwords.words("english"))
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
def FilterQuestions(arrstrQuestions):
    for i, strQuestion in enumerate(arrstrQuestions):
        punctuationlessQuestion = RemovePunctuationCharacters(strQuestion) 
        punctuationlessQuestion = ' '.join([word for word in punctuationlessQuestion.split() if word not in cachedStopWords])
        punctuationlessQuestion = " ".join([lemmatizer.lemmatize(i) for i in punctuationlessQuestion.split()])
        arrstrQuestions[i] = punctuationlessQuestion
        
    return arrstrQuestions

def learnModel(data):
    if os.path.isfile("BagOfWordsSVMNauceni.pkl"):
        return None
    data[0] = FilterQuestions(data[0])
    data[1] = FilterQuestions(data[1])
    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.  
    vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 20000) 
    
    allQuestions = data[0] + data[1]

    vectorizer.fit(allQuestions)    
    joblib.dump(vectorizer, 'BagOfWordsVectorizerNauceni.pkl') 
    
    znacajkePitanja = [vectorizer.transform(data[0]), vectorizer.transform(data[1])]
    for i, r in enumerate(data[2]):
        data[2][i] = int(r)
        
    znacajkePitanja = hstack(znacajkePitanja).tocsr()
    svmKlasifikator = SVC(kernel='rbf', verbose=True, probability=True, max_iter=1000000)
   
    print("Learning started")
    tmStart = timer()
    svmKlasifikator.fit(znacajkePitanja, data[2])
    tmEnd = timer()
    print("Learning ended")
    print("Learning lasted", tmEnd - tmStart)
    
    joblib.dump(svmKlasifikator, 'BagOfWordsSVMNauceni.pkl') 
    print("Spremljen je napredak ucenja")

def loadData(strFileName):
    data = pd.read_csv(strFileName, sep=',', lineterminator="\n")
    dataTable = []
    
    for j, c in enumerate(data):
        dataTable.append([])
        for cell in data[c]:
            dataTable[j].append(cell)
        
    return dataTable

def predict(data, strOutputfile):
    svmModel = joblib.load('BagOfWordsSVMNauceni.pkl')
    vectorizer = joblib.load('BagOfWordsVectorizerNauceni.pkl')
    data[0] = FilterQuestions(data[0])
    data[1] = FilterQuestions(data[1])
    
    znacajkePitanja = [vectorizer.transform(data[0]), vectorizer.transform(data[1])]    
    znacajkePitanja = hstack(znacajkePitanja).tocsr()
    print("Predicting started")
    tmStart = timer()
    prediction = svmModel.predict(znacajkePitanja)
    tmEnd = timer()
    print("Predicting ended")
    print("Predicting lasted", tmEnd - tmStart)
    
    fileOutout = open(strOutputfile, "w")
    fileOutout.write("test_id,is_duplicate\n")
    for i, p in enumerate(prediction):
        fileOutout.write(",".join([str(i), str(p)])+"\n")
        
    print("Prediction saved: Done")

learningTable = loadData('train.csv')
learnModel(learningTable[3:6])
testTable = loadData('test.csv')
predictionData = [testTable[1], testTable[2]]
predict(predictionData, 'predikcija.out')