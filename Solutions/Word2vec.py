import os
import pandas as pd
import re
import string

import nltk.data
nltk.download()

from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def RemovePunctuationCharacters(strQuestion):
    strRetval = str(strQuestion)
    for char in string.punctuation:
        strRetval = strRetval.replace(char, "")
        
    strRetval = re.sub(' +', ' ', strRetval)
    return strRetval.lower().split()

train = pd.read_csv('train.csv', sep=',', header=0, lineterminator="\n")

print(RemovePunctuationCharacters(train["question1"][58]))

for i in range (0, len(train)):
    RemovePunctuationCharacters(train["question1"][i])
    RemovePunctuationCharacters(train["question2"][i])
   
    



