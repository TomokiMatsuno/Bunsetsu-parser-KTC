import pandas as pd
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors

def get_pret_embs():
    path2javec = '/Users/tomoki/NLP_data/ja-vec-w2v-format.txt'

    word_vectors = KeyedVectors.load_word2vec_format(path2javec)

    return word_vectors

# for i in range(len(word_vectors.index2word)):
#     print(word_vectors.index2word[i])
