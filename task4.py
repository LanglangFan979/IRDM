import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from math import log

#import test queries
test_queries = np.asarray(pd.read_csv('test-queries.tsv', delimiter= '\t', header = None))
test_ind_dict = {qid: index for index,qid in enumerate(test_queries[:,0])}

#load candidate passages
candidate_passage = np.asarray(pd.read_csv('candidate-passages-top1000.tsv', delimiter= '\t', header = None))

def remove_stopwords(tokens):
    new_tokens = []
    stop_words = set(stopwords.words("english"))
    for token in tokens:
        if token not in stop_words:
            new_tokens.append(token)
    return new_tokens

def preprocess(documents):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(documents)
    tokens =  [token.lower() for token in tokens if token.isalpha()]
    tokens = remove_stopwords(tokens)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token, pos='v') for token in tokens]
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

V = len(inverted_index)

def Laplace_smooth(passage,V):
    tokens = preprocess(passage)
    uniq_tokens, freq = np.unique(tokens, return_counts= True)
    length = np.sum(freq)+V
    freq = freq+1
    dist = dict(zip(uniq_tokens, np.log(freq/length)))
    return dist, log(1/length)

def Lindstone_correction(passage, V, epsilon = 0.1):
    tokens = preprocess(passage)
    uniq_tokens, freq = np.unique(tokens, return_counts=True)
    doc_len = freq.sum()
    length = doc_len + epsilon*V
    freq = freq+epsilon
    dist = dict(zip(uniq_tokens, np.log(freq/length)))
    return dist, log(epsilon/length)

def query_likelihood(contents, smoothing, V):
    scores = np.zeros((len(contents),3))
    for i in range(len(contents)):
        tokens_q = set(preprocess(contents[i,-2]))
        passage = contents[i,-1]
        if smoothing == Laplace_smooth:
            passage_dist, notin = smoothing(passage, V)
        elif smoothing == Lindstone_correction:
            passage_dist, notin = smoothing(passage, V,epsilon = 0.1)
        likelihood = 0
        for token in tokens_q:
            if token not in passage_dist.keys():
                likelihood += notin
            else:
                likelihood += passage_dist[token]
        scores[i,:2], scores[i,-1] = contents[i,:2].astype('int'), likelihood
    return scores

def Dirichlet_smooth_likelihood(contents, inverted_index,collection_len, miu = 50):
    scores = np.zeros((len(contents),3))
    for i in range(len(contents)):
        tokens_q = set(preprocess(contents[i,-2]))
        passage = contents[i,-1]
        tokens = preprocess(passage)
        freq_p_dist = nltk.FreqDist(tokens)
        doc_len = sum(freq_p_dist.values())
        coeff_doc = doc_len/(doc_len + miu)
        coeff_col = miu/(doc_len + miu)
        likelihood = 0
        for token in tokens_q:
            if token not in freq_p_dist.keys():
                if token not in inverted_index.keys():
                    likelihood += 0
                if token in inverted_index.keys():
                    likelihood += log((inverted_index[token][1]/collection_len)*coeff_col)
            else:
                #document part
                freq_inp = freq_p_dist[token]/doc_len
                #collection part
                freq_inc = inverted_index[token][1]/collection_len
                #distribution
                likelihood += log(freq_inp*coeff_doc + freq_inc*coeff_col)
        scores[i,:2], scores[i,-1] = contents[i,:2].astype('int'), likelihood
    return scores

def top100(scores, test_ind_dict):
    top = np.zeros(3)
    for qid,ind in test_ind_dict.items():
        candidates = scores[np.where(scores[:,0] == qid)]
        if len(candidates) >100:
            rank = np.argsort(candidates[:,-1])[::-1][:100]
        else:
            rank = np.argsort(candidates[:,-1])[::-1][:100]
        top = np.vstack((top, candidates[rank]))
    return top[1:,:]

laplace_scores = query_likelihood(candidate_passage,Laplace_smooth, V)
laplace = top100(laplace_scores, test_ind_dict)
np.savetxt("laplace.csv", laplace, fmt='%d,%d,%s', delimiter=',')

Lindstone_scores = query_likelihood(candidate_passage,Lindstone_correction, V)
lindstone = top100(Lindstone_scores, test_ind_dict)
np.savetxt("lidstone.csv", lindstone, fmt='%d,%d,%s', delimiter=',')

collection_len = 0
for value in inverted_index.values():
    collection_len += value[1]

Dirichlet_scores = Dirichlet_smooth_likelihood(candidate_passage, inverted_index,collection_len)
dirichlet = top100(Dirichlet_scores, test_ind_dict)
np.savetxt("dirichlet.csv", dirichlet, fmt='%d,%d,%s', delimiter=',')

