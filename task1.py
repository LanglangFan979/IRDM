import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
# from nltk.stem import WordNetLemmatizer
# from nltk import WordPunctTokenizer
import matplotlib.pyplot as plt

#import data
with open('passage-collection.txt') as passages:
    contents = passages.read()
passages.close()

def preprocess(documents):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(documents)
    tokens =  [token.lower() for token in tokens if token.isalpha()]
    return tokens

tokens = preprocess(contents)
#vocabulary
vocabulary, frequency = np.unique(np.asarray(tokens), return_counts= True)
print(len(vocabulary))#report size of the identified index of terms

#rank the frequencies and list vocabulary according to rank
rank = np.argsort(frequency)[::-1] #indices of ranked frequency
vocabulary_ranked = vocabulary[rank]
frequency_ranked = frequency[rank]

#compute the normilazed frequency
normalized_frequency = frequency_ranked/(frequency.sum())

#plot probability against frequency ranking
plt.plot(np.arange(1, len(vocabulary)+1), normalized_frequency)#the rank above is indices not rank
plt.xlabel('Term frequency ranking')
plt.ylabel('Term prob. of occurence')
plt.savefig("D1_1")
plt.show()

#standard zipf's distribution
zipf = 1/(np.arange(1, len(vocabulary)+1 )*np.sum(1/np.arange(1, len(vocabulary) +1)))

#plot standard Zipf's distribution
plt.plot(np.log(np.arange(1, len(vocabulary)+1)),np.log(normalized_frequency), label='Empirical distribution(log)')
plt.plot(np.log(np.arange(1, len(vocabulary)+1)),np.log(zipf), label="Zipf's distribution(log)")
plt.xlabel('Term frequency ranking(log)')
plt.ylabel('Term prob. of occurence (log)')
plt.legend()
plt.savefig("D1_2")
plt.show()



