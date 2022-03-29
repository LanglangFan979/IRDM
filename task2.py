import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

#import 'candidate-passage-top1000.tsv' data
candidate_passage = np.asarray(pd.read_csv('candidate-passages-top1000.tsv', delimiter= '\t', header = None))

def remove_stopwords(tokens):
    new_tokens = []
    stop_words = set(stopwords.words("english"))
    for token in tokens:
        if token not in stop_words:
            new_tokens.append(token)
    return new_tokens

def lemmatization(tokens):
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token, pos='v') for token in tokens]
    return tokens

def preprocess(documents):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(documents)
    tokens =  [token.lower() for token in tokens if token.isalpha()]
    tokens = remove_stopwords(tokens)
    tokens = lemmatization(tokens)
    return tokens

passages, ind = np.unique(candidate_passage[:,1], return_index= True)
inverted_index = {}
for i in range(len(ind)):
    tokens = preprocess(candidate_passage[ind[i],-1])
    tokens_dist = nltk.FreqDist(tokens)
    for voc in tokens_dist.keys():
        if voc not in inverted_index.keys():
            inverted_index[voc] = [1, tokens_dist[voc]]
        else:
            inverted_index[voc][0] += 1
            inverted_index[voc][1] += tokens_dist[voc]