import os
import pandas as pd
import re
import string
from timeit import default_timer as timer

import nltk.data
#nltk.download()

from gensim.parsing.preprocessing import remove_stopwords

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import numpy as np

from sklearn.externals import joblib
from sklearn.svm.classes import SVC
from nltk.tbl import feature
from scipy.sparse import coo_matrix, hstack

def RemovePunctuationCharacters( strQuestion, remove_stopwords = False ):
    strRetval = str(strQuestion)
    for char in string.punctuation:
        strRetval = strRetval.replace(char, "")
        
    strRetval = re.sub(' +', ' ', strRetval)
    
    words = strRetval.lower().split()
    
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    return words

def question_to_sentences( question, remove_stopwords = False ):
    sentences = []
    sentences.append( RemovePunctuationCharacters(question, remove_stopwords) )
    return sentences

def learn_w2v ( train ):
    
    question1 = []
    question2 = []

    for question in train["question1"]:
        question1 += question_to_sentences(question, False)
    
    for question in train["question2"]:
        question2 += question_to_sentences(question, False)
        
    sentences = question1 + question2
    # Set values for various parameters
    num_features = 300    # Word vector dimensionality                      
    min_word_count = 1   # Minimum word count                        
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size                                                                                    
    downsampling = 1e-4   # Downsample setting for frequent words
    
    import logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

    from gensim.models import word2vec

    print ("Training model...")
    model = word2vec.Word2Vec(sentences, sg=1, workers=num_workers, \
                size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling, hs=1, negative=0)
    
    # If you don't plan to train the model any further, calling 
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)
    
    # It can be helpful to create a meaningful model name and 
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "300features_40minwords_10context"
    model.save(model_name)
    print ("Gotovo")
    
def addLists(v1, v2):
    sum = [x + y for x, y in zip(v1, v2)]
    if not len(v1) >= len(v2):
        sum += v2[len(v1):]
    else:
        sum += v1[len(v2):]

    return sum
    
def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given paragraph
    # Pre-initialize an empty numpy array (for speed)
    #featureVec = np.zeros((num_features,),dtype="float32")
    featureVec = [0]*num_features
    nwords = 0
     
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.wv.index2word)
    
    # Loop over each word in the review and, if it is in the model's vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1
            #featureVec = np.add(featureVec,model[word])
            featureVec = addLists(featureVec, model[word])
     
    # Divide the result by the number of words to get the average
    if nwords > 0:
        #featureVec = np.divide(featureVec,nwords)
        featureVec = [i/nwords for i in featureVec]
    return featureVec


def getAvgFeatureVecs(questions, model, num_features):
    # Given a set of questions (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 

    counter = 0 
    # Preallocate a 2D numpy array, for speed
    #questionFeatureVecs = np.zeros((len(questions),num_features),dtype="float32")
    questionFeatureVecs = []
    for i in range(0, len(questions)):
        questionFeatureVecs.append([0]*num_features)

    for question in questions:
        # Print a status message every 1000th question
        if counter%1000 == 0:
            print ("Question %d of %d" % (counter, len(questions))) 
        # Call the function (defined above) that makes average feature vectors
        questionFeatureVecs[counter] = makeFeatureVec(question, model, num_features)

        counter = counter + 1
    return questionFeatureVecs

def get_avg( data ):
    data_sentences1 = []  
    data_sentences2 = []
    num_features = 300
    #Tu je taj brojač i, a vezan je za iduće dvije petlje
    i = 0

    print ("Parsing sentences from training set")
    for question in data["question1"]:
        if i < 100:
            data_sentences1 += question_to_sentences( question, remove_stopwords = True )
        else: exit
        i += 1
    
    i = 0
    for question in data["question2"]:
        if i < 100:
            data_sentences2 += question_to_sentences( question, remove_stopwords = True )
        else: exit
        i += 1

    from gensim.models import Word2Vec
    model = Word2Vec.load("300features_40minwords_10context")

    DataVecs1 = coo_matrix(getAvgFeatureVecs( data_sentences1, model, num_features ))
    DataVecs2 = coo_matrix(getAvgFeatureVecs( data_sentences2, model, num_features ))
    
    DataVecs = hstack([DataVecs1, DataVecs2]).toarray()

    return DataVecs

    
def learnModel( train ):

    data = []
    for duplicate in train["is_duplicate"]:
        data.append( int(duplicate))
        
    znacajkePitanja = get_avg(train)
    svmKlasifikator = SVC(kernel='rbf', verbose=True, probability=True, max_iter=10000)
   
    print("Learning started")
    tmStart = timer()
    svmKlasifikator.fit(znacajkePitanja, data)
    tmEnd = timer()
    print("Learning ended")
    print("Learning lasted", tmEnd - tmStart)
    
    joblib.dump(svmKlasifikator, 'Word2VecSVMNauceni.pkl') 
    print("Spremljen je napredak ucenja")
    
def predict(data, strOutputfile):
    svmModel = joblib.load('Word2VecSVMNauceni.pkl')
    
    znacajkePitanja = get_avg( data )
    
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

#train = pd.read_csv('train.csv', sep=',', header=0, lineterminator="\n")
test = pd.read_csv('test.csv', sep=',', header=0, lineterminator="\n")

#learn_w2v(train)
#learnModel( train )
predict(test, 'w2vPredikcija.out')
