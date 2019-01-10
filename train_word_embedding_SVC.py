from collections import defaultdict
from os.path import join
import numpy as np
import gensim
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import dill

def SaveModel(clf):
    filename = 'QUALITY1.pkl'
    saved_model = open(join("models_mean_embedding_SVC", filename), 'wb')
    dill.dump(clf, saved_model)
    saved_model.close()


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        # self.dim = len(word2vec.itervalues().next())
        # self.dim = len(next(word2vec.values()))
        self.dim = 100

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        # self.dim = len(word2vec.itervalues().next())
        # self.dim = len(next(iter(word2vec.values())))
        self.dim = 100

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])

datas = []
categories = []

with open(join("data_train", "datas_QUALITY.txt"), 'r', encoding='utf-8')as file:
    for i in file:
        datas.append(i)

with open(join("data_train", "labels_QUALITY.txt"), 'r', encoding='utf-8')as file:
    for i in file:
        categories.append(i)

df = pd.DataFrame({"datas": datas, "categories": categories})
X_train = df['datas']
y_train = df['categories']

model = gensim.models.KeyedVectors.load_word2vec_format('w2v_bao_100_mincount3_new.bin', binary=True,
                                                        encoding='utf8')

print("Similarity between 'đỏ' and 'đen'",
      model.wv.similarity(w1="đỏ", w2="đen"))

with open("w2v_bao_100_mincount3_new.txt", "r", encoding='utf-8') as lines:
    w2v = {line.split()[0]: np.array(map(float, line.split()[1:]))
           for line in lines}

print(len(w2v))

SVC_w2v_mean_embedding = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
                          ('clf', SVC(kernel='linear')),
                          ])

SVC_w2v_mean_embedding.fit(X_train, y_train)
SaveModel(SVC_w2v_mean_embedding)

