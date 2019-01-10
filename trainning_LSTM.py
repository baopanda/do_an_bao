import pickle
from os.path import join
import pandas as pd
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def SaveModel(clf):
    filename = 'AMBIENCE.pkl'
    saved_model = open(join("models_RF", filename), 'wb')
    pickle.dump(clf, saved_model)
    saved_model.close()


datas = []
labels = []
datas_valid = []
labels_valid = []

with open(join("data_train", "datas_AMBIENCE.txt"), 'r', encoding='utf-8')as file:
    for i in file:
        datas.append(i)

with open(join("data_train", "labels_AMBIENCE.txt"), 'r', encoding='utf-8')as file:
    for i in file:
        if(i == 'None\n'):
            labels.append(0)
        else:
            labels.append(1)

with open(join("data_test", "datas_AMBIENCE.txt"), 'r', encoding='utf-8')as file:
    for i in file:
        datas_valid.append(i)
with open(join("data_test", "labels_AMBIENCE.txt"), 'r', encoding='utf-8')as file:
    for i in file:
        if (i == 'None\n'):
            labels_valid.append(0)
        else:
            labels_valid.append(1)


df = pd.DataFrame({"datas": datas, "categories": labels})
df1 = pd.DataFrame({"datas": datas_valid, "categories": labels_valid})
data = df['datas']
label = df['categories']
data_valid = df1['datas']
label_valid = df1['categories']
print(df.describe())
print(data.shape)
print(data_valid.shape)
print(label.shape)
print(label_valid.shape)


# vocabulary_size = 20000
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['datas'])
sequences = tokenizer.texts_to_sequences(df['datas'])
data = pad_sequences(sequences)
print(data.shape)
#
# vocabulary_size = 20000
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df1['datas'])
sequences1 = tokenizer.texts_to_sequences(df1['datas'])
data_valid = pad_sequences(sequences1)
print(data_valid.shape)



model_lstm = Sequential()
model_lstm.add(Embedding(20000, 100))
model_lstm.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model_lstm.add(Dense(1, activation='sigmoid'))
model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model=Sequential()
model.add(Embedding(20000,100))
model.add(Dropout(0.2))
model.add(Conv1D(64, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(data,label, validation_data=(data_valid,label_valid), epochs=100)