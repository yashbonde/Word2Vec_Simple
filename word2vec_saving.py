# Simple saver for word2vec

# importing the dependencoes
import numpy as np # matrix math
import nltk # NLP
import re # NLP
import gensim.models.word2vec as word2vec # word2vec model
import multiprocessing # cpu_count
import os # for storing and retirieving the files
import codecs # for encoding suring opening

# open the file and read the lines
file_path = '~/data/chat.txt'
f = codecs.open(file_path, 'r', 'utf-8')
corpus = u""
corpus += f.read()

# tokenising the text
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentencesList = tokenizer.tokenize(corpus)

def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words

#sentence where each word is tokenized
sentences = []
for raw_sentence in sentencesList:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))

# params for word2vec model
e_dim = 100 # dimension of the vector that we want
workers = multiprocessing.cpu_count() # for multprocessing
min_word_count = 1 # minimum number of times a word must come to be registered
# I have used 1 because I have to further use it for embedding in an RNN
context_size = 8 # length of context sentence that would be considered
downsample = 1e-5 # for words that come too frequently

# Defining the model
w2vector = word2vec.Word2Vec(sg=1, seed=1, size = e_dim, workers = workers,
                             min_count = min_word_count, window = context_size,
                             sample = downsample)

# building the vocabulary
w2vector.build_vocab(sentences)

# Training the model
w2vector.train(sentences, total_examples = w2vector.corpus_count, epochs = w2vector.iter)

# saving the model, so that we can use them laters
if not os._exists('trained'):
    os.makedirs('trained')

w2vector.save(os.path.join('trained', 'w2vector.w2v'))