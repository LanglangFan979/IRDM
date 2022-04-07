import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from math import log10, sqrt,log

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

def cosine_similarity(contents, preprocess, inverted_index):
    scores = np.zeros((len(contents),3))
    N = len(set(contents[:,1]))
    for i in range(len(contents)):
        nominator = 0
        tokens_p = preprocess(contents[i,-1])
        tokens_q = preprocess(contents[i,-2])
        freq_p = nltk.FreqDist(tokens_p)
        freq_q = nltk.FreqDist(tokens_q)
        p_len = len(tokens_p)
        q_len = len(tokens_q)
        deno_p = 0
        for token in freq_p.keys():
            deno_p += (log10(N/inverted_index[token][0])*(freq_p[token]/p_len))**2
        deno_p = sqrt(deno_p)
        deno_q = 0
        for token,freq in freq_q.items():
            if token in inverted_index.keys():
                deno_q += (log10(N/inverted_index[token][0])*(freq_q[token]/q_len))**2
            if token in freq_p.keys():
                nominator += (log10(N/inverted_index[token][0])**2)*(freq_p[token]/p_len) *(freq_q[token]/q_len)
        deno_q = sqrt(deno_q)
        scores[i,0], scores[i,1], scores[i,-1] = int(contents[i,0]), int(contents[i,1]), (nominator/(deno_q*deno_p))
    return scores

cosine_socres = cosine_similarity(candidate_passage, preprocess,inverted_index)

tfidf = np.zeros(3)
for qid,ind in test_ind_dict.items():
    scores = cosine_socres[np.where(cosine_socres[:,0] == qid)]
    if len(scores) >100:
        top100 = np.argsort(scores[:,-1])[::-1][:100]
    else:
        top100 = np.argsort(scores[:,-1])[::-1][:100]
    tfidf = np.vstack((tfidf, scores[top100]))
tfidf = tfidf[1:,:]

np.savetxt("tfidf.csv", tfidf, fmt='%d,%d,%s', delimiter=',')

def length(contents):
    total_len = 0
    unique_pa = np.unique(contents[:,-1])
    for i in range(len(unique_pa)):
        passage_len = len(preprocess(unique_pa[i]))
        total_len += passage_len
    total_pa = len(unique_pa)
    return total_len/total_pa, total_pa

avdl, N = length(candidate_passage)

def BM25(contents, inverted_index,avdl,N, k1 = 1.2, k2 = 100,b = 0.75):
    scores = np.zeros((len(contents),3))
    for i in range(len(contents)):
        qid = contents[i,0]
        tokens_p = preprocess(contents[i,-1])
        tokens_q = preprocess(contents[i,-2])
        f_p = nltk.FreqDist(tokens_p)
        f_q = nltk.FreqDist(tokens_q)
        dl = len(tokens_p)
        K = k1*((1-b)+b*(dl/avdl))
        bm25 = 0
        for token in f_q.keys():
            if token in inverted_index.keys():
                term1 =  log((N-inverted_index[token][0]+0.5)/(inverted_index[token][0]+0.5))
                term2 = (k1+1)*f_p[token]/(K+f_p[token])
                term3 = (k2+1)*f_q[token]/(k2+f_q[token])
                bm25 += term1 *term2 *term3
        scores[i,:2], scores[i,-1] = contents[i,:2].astype('int'), bm25
    return scores[1:,:]

bm25_scores = BM25(candidate_passage, inverted_index, avdl, N)

bm25 = np.zeros(3)
for qid,ind in test_ind_dict.items():
    scores = bm25_scores[np.where(bm25_scores[:,0] == qid)]
    if len(scores) >100:
        top100 = np.argsort(scores[:,-1])[::-1][:100]
    else:
        top100 = np.argsort(scores[:,-1])[::-1][:100]
    bm25 = np.vstack((bm25, scores[top100]))
bm25 = bm25[1:,:]

np.savetxt("bm25.csv", bm25,  fmt='%d,%d,%s', delimiter=',')



