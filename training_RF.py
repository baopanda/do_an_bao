import pickle
from os.path import join
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC



def SaveModel(clf):
    filename = 'SERVICE_new.pkl'
    saved_model = open(join("models_RF_new",filename), 'wb')
    pickle.dump(clf, saved_model)
    saved_model.close()

datas = []
categories = []
datas_valid = []
categories_valid = []

with open(join("data_train", "datas_SERVICE_new.txt"),'r', encoding='utf-8')as file:
    for i in file:
        datas.append(i)

with open(join("data_train", "labels_SERVICE_new.txt"),'r', encoding='utf-8')as file:
    for i in file:
        categories.append(i)


df = pd.DataFrame({"datas": datas, "categories": categories})
data = df['datas']
label = df['categories']

X_train = datas
y_train = label

text_clf_svm = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', RandomForestClassifier()), ])

print(text_clf_svm.get_params().keys())
param_grid = {
    "clf__max_features": np.linspace(0.2, 1, 5),
    "clf__n_estimators": np.arange(50, 150, 30),
    "clf__max_depth": np.arange(30, 80, 20)
}
text_clf_svm = GridSearchCV(text_clf_svm, param_grid=param_grid, cv=5)
text_clf_svm = text_clf_svm.fit(X_train, y_train)
print(text_clf_svm)

SaveModel(text_clf_svm)

