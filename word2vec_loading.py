# Simple loader for word2vec

# importing the dependencoes
import numpy as np # matrix math
import nltk # NLP
import re # NLP
import gensim.models.word2vec as word2vec # word2vec model
import multiprocessing # cpu_count
import os # for storing and retirieving the files
import codecs # for encoding suring opening

# loading the model
w2vector = word2vec.Word2Vec.load(os.path.join('trained', 'w2vector.w2v'))

# now to get any embedding value
x = w2vector['drink']
print(x)

# this will give a numpy array of size [emedding_dimension,]