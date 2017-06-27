import pandas as pd
from gensim.models import Doc2Vec
import multiprocessing
import string
import re
from timeit import default_timer as timer
from sklearn.svm.classes import SVC
from sklearn.externals import joblib
import os

iCores = multiprocessing.cpu_count()

print("Broj jezgri: " + str(iCores))

def RemovePunctuationCharacters(strQuestion):
    strRetval = str(strQuestion)
    for char in string.punctuation:
        strRetval = strRetval.replace(char, "")
        
    strRetval = re.sub(' +', ' ', strRetval)
    return strRetval.lower()

def FilterQuestions(arrstrQuestions):
    for i, strQuestion in enumerate(arrstrQuestions):
        arrstrQuestions[i] = RemovePunctuationCharacters(strQuestion)
    return arrstrQuestions


def loadData(strFileName):
    data = pd.read_csv(strFileName, sep=',', lineterminator="\n")
    table = [[y for y in data[x]] for x in data]
    return table

def prepareDoc2Vec(aQ1, aQ2):
    class Pitanje:
        def __init__(self, strPitanje):
            self.words = strPitanje.split(" ")
            self.tags = [0]
    svaPitanja = aQ1 + aQ2
    svaPitanja = [Pitanje(x) for x in svaPitanja]
    print("instancijacija")
    model = Doc2Vec(svaPitanja, size = 100, window = 5, min_count = 5, workers = iCores)
    print("treniram doc2vec")
    tmStart = timer()
    model.train(svaPitanja, total_examples=len(svaPitanja), epochs=200)
    tmEnd = timer()
    print("treniranje doc2vec trajalo", tmEnd - tmStart)
    model.save("Doc2VecModel.d2v")
    joblib.dump(model, "Doc2VecModelNauceni.pkl")
    return model

def learnPhase():
    if os.path.isfile("Doc2VecSVMNauceni.pkl"):
        return None
    tablecolrow = loadData("train.csv")
    tablecolrow[3] = FilterQuestions(tablecolrow[3])
    tablecolrow[4] = FilterQuestions(tablecolrow[4])
    
    model = prepareDoc2Vec(tablecolrow[3], tablecolrow[4])
    
    for i in range(len(tablecolrow[3])):
        tablecolrow[3][i] = model.infer_vector(tablecolrow[3][i].split(" "))
        tablecolrow[4][i] = model.infer_vector(tablecolrow[4][i].split(" "))
    
    traindataX = [None] * len(tablecolrow[3])
    traindataY = [None] * len(tablecolrow[3])
    for i in range(len(traindataX)):
        traindataX[i] = tablecolrow[3][i] + tablecolrow[4][i]
        traindataY[i] = int(tablecolrow[5][i])
        
    svmKlasifikator = SVC(kernel='rbf', verbose=True, probability=True, max_iter=1000000)
    print("Learning started")
    tmStart = timer()
    svmKlasifikator.fit(traindataX, traindataY)
    tmEnd = timer()
    print("Predicting lasted", tmEnd - tmStart)
    joblib.dump(svmKlasifikator, 'Doc2VecSVMNauceni.pkl') 
    print("Spremljen je napredak ucenja")
    
def testingPhase():
    print("test started")
    tablecolrow = loadData("test.csv")
    tablecolrow[1] = FilterQuestions(tablecolrow[1])
    tablecolrow[2] = FilterQuestions(tablecolrow[2])
    model = joblib.load('Doc2VecModelNauceni.pkl')
    svmModel = joblib.load('Doc2VecSVMNauceni.pkl')
    print("fase start")
    for i in range(len(tablecolrow[1])):
        tablecolrow[1][i] = model.infer_vector(tablecolrow[1][i].split(" "))
        tablecolrow[2][i] = model.infer_vector(tablecolrow[2][i].split(" "))
    
    traindataX = [None] * len(tablecolrow[1])
    for i in range(len(traindataX)):
        traindataX[i] = tablecolrow[1][i] + tablecolrow[2][i]
    
    print("Predicting started")
    tmStart = timer()
    prediction = svmModel.predict(traindataX)
    tmEnd = timer()
    print("Predicting ended")
    print("Predicting lasted", tmEnd - tmStart)
    
    fileOutout = open("doc2vecPredikcija.out", "w")
    fileOutout.write("test_id,is_duplicate\n")
    for i, p in enumerate(prediction):
        fileOutout.write(",".join([str(i), str(p)])+"\n")
        
    print("Prediction saved: Done")
    
learnPhase()
testingPhase()