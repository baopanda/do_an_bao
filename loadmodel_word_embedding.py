import pickle
from os.path import join

from pyvi import ViTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report
import PreProcessing_valid
import builtins
import dill
import numpy as np



def LoadData(path_data,path_label):
    datas = []
    labels = []
    with open(path_data, 'r', encoding='utf-8')as file:
        for i in file:
            datas.append(i)

    with open(path_label, 'r', encoding='utf-8')as file:
        for i in file:
            labels.append(i)
    return datas, labels

def Classification():
    s = "Đồ ăn tại quán ăn rất ngon"
    # s= "Mình thấy suất XL ở đây to hơn, ngon hơn và rất đẹp, nhưng rất đắt"
    s = PreProcessing_valid.PreProcessing(s)
    print(s)
    pre = []
    pre.append(s)
    datas_valid = []
    labels_valid = []
    # vectorizer = CountVectorizer()
    # transformed_x_valid = vectorizer.fit_transform(s).toarray()
    load_file = open(join("models_word_embedding","QUALITY.pkl"),'rb')
    clf = dill.load(load_file)
    # print("Loading file : ",clf)

    with open(join("data_test", "datas_QUALITY.txt"), 'r', encoding='utf-8')as file:
        for i in file:
            datas_valid.append(i)
    with open(join("data_test", "labels_QUALITY.txt"), 'r', encoding='utf-8')as file:
        for i in file:
            labels_valid.append(i)

    X_valid = datas_valid
    a = clf.predict(X_valid)
    # with open("predict.txt",'w',encoding='utf-8') as f:
    #     for i in a:
    #         f.write(i)
    t = clf.predict(pre)
    print(t)
    print(a)
    print(confusion_matrix(labels_valid, a))
    # print(classification_report(labels_valid,a))



if __name__ == "__main__":
    Classification()

