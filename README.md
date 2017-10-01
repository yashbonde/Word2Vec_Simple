# Word2Vec_Simple
This is a simple usage of word to vector model wich was created by a team of researchers led by Tomas Mikolov at Google.

Embedding refers to converting a single number into a vector which can be fed into an RNN.

Example:- [3] --> [0.2, 0.5, -0.4] which is a vector of embedding dimension 3.

Word Embedding comes to play when a vector of numbers which is a the normal way of representation of text, is converted into a matrix of size [total_words_in_sentences, embedding_dimension].

Word2Vector is one such model for implementation Word Embedding. The word embedding approach is able to capture multiple different degrees of similarity between words. It is found that semantic and syntactic patterns can be reproduced using vector arithmetic. Patterns such as “Man is to Woman as Brother is to Sister” can be generated through algebraic operations on the vector representations of these words such that the vector representation of “Brother” - ”Man” + ”Woman” produces a result which is closest to the vector representation of “Sister” in the model. Such relationships can be generated for a range of semantic relations (such as Country—Capital) as well as syntactic relations (e.g. present tense—past tense)

for more details read their wikipedia page: https://en.wikipedia.org/wiki/Word2vec
