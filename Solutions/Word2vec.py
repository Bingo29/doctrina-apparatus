import os
import pandas as pd
import re
import string

import nltk.data
#nltk.download()

from nltk.stem import WordNetLemmatizer
from gensim.parsing.preprocessing import remove_stopwords
lemmatizer = WordNetLemmatizer()

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import numpy as np
from scipy.sparse import hstack
import scipy.sparse

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

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

def learn_w2v ( sentences ):
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
    #model.init_sims(replace=True)
    
    # It can be helpful to create a meaningful model name and 
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "300features_40minwords_10context"
    model.save(model_name)
    print ("Gotovo")

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given paragraph
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0
     
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.wv.index2word)
    
    # Loop over each word in the review and, if it is in the model's vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word])
     
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(questions, model, num_features):
    # Given a set of questions (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 

    counter = 0 
    # Preallocate a 2D numpy array, for speed
    questionFeatureVecs = np.zeros((len(questions),num_features),dtype="float32")

    for question in questions:
        # Print a status message every 1000th question
        if counter%1000 == 0:
            print ("Question %d of %d" % (counter, len(questions))) 
        # Call the function (defined above) that makes average feature vectors
        questionFeatureVecs[counter] = makeFeatureVec(question, model, num_features)

        counter = counter + 1
    return questionFeatureVecs

def get_avg( train ):
    train_sentences1 = []  
    train_sentences2 = []
    num_features = 300
    i = 0

    print ("Parsing sentences from training set")
    for question in train["question1"]:
        i += 1
        if i<20:
            train_sentences1 += question_to_sentences( question, remove_stopwords = True )
        else: exit

    i = 0
    for question in train["question2"]:
        i += 1
        if i<20:
            train_sentences2 += question_to_sentences( question, remove_stopwords = True )
        else: exit
        
    #train_allQuestions = train_sentences1 + train_sentences2
       
    #learn_w2v(allQuestions)
    from gensim.models import Word2Vec
    model = Word2Vec.load("300features_40minwords_10context")
    
    trainDataVecs1 = getAvgFeatureVecs( train_sentences1, model, num_features )
    trainDataVecs2 = getAvgFeatureVecs( train_sentences2, model, num_features )
    
    #U OVOJ SLJEDEĆOJ LINIJI JE PROBLEM!!!
    trainDataVecs = np.asarray([trainDataVecs1, trainDataVecs2])
    #trainDataVecs = hstack(trainDataVecs).tocsr

    
    return trainDataVecs


train = pd.read_csv('train.csv', sep=',', header=0, lineterminator="\n")
test = pd.read_csv('test.csv', sep=',', header=0, lineterminator="\n")

data = []

for duplicate in train["is_duplicate"]:
    data.append( int(duplicate))

trainDataVecs = get_avg( train )
testDataVecs = get_avg( test )

# Fit a random forest to the training data, using 100 trees
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier( n_estimators = 100 )

print ("Fitting a random forest to labeled training data...")
forest = forest.fit( trainDataVecs, data )

result = forest.predict( testDataVecs )

# Write the test results 
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output.to_csv( "Word2Vec_AverageVectors.csv", index=False, quoting=3 )

print("Gotovo")
