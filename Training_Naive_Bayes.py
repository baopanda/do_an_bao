import pickle
from os.path import join
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

def SaveModel(clf):
    filename = 'STYLEOPTIONS_new.pkl'
    saved_model = open(join("models_NB",filename), 'wb')
    pickle.dump(clf, saved_model)
    saved_model.close()

datas = []
categories = []
datas_valid = []
categories_valid = []

with open(join("data_train", "datas_STYLEOPTIONS_new.txt"),'r', encoding='utf-8')as file:
    for i in file:
        datas.append(i)

with open(join("data_train", "labels_STYLEOPTIONS_new.txt"),'r', encoding='utf-8')as file:
    for i in file:
        categories.append(i)


df = pd.DataFrame({"datas": datas, "categories": categories})
data = df['datas']
label = df['categories']

X_train = datas
y_train = label

# text_clf_NB = Pipeline([('vect', CountVectorizer()),
#                          ('tfidf', TfidfTransformer()),
#                          ('clf', MultinomialNB()), ])

text_clf_NB = Pipeline([('vect', CountVectorizer()),
                         # ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB()),
                        ])

print(text_clf_NB.get_params().keys())
param_grid = {
    "clf__alpha": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    # 'vect__ngram_range': ((1, 1), (1, 2)),
    # 'tfidf__use_idf': (False)
}
text_clf_svm = GridSearchCV(text_clf_NB, param_grid=param_grid, cv=5)
text_clf_svm = text_clf_svm.fit(X_train, y_train)
print(text_clf_svm)

SaveModel(text_clf_svm)

