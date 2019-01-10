import pickle
from os.path import join

from pyvi import ViTokenizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,f1_score,recall_score
import PreProcessing_valid
from lxml import etree as ET

import StopWords

def Load_data(name_cate):
    tree = ET.parse(join("data", "data_test_doan.xml"))
    root = tree.getroot()
    datas_valid1 = []
    labels_valid1 = []
    reviews = root.findall("Review")
    sentences = root.findall("**/sentence")
    print("# Reviews   : ", len(reviews))
    print("# Sentences : ", len(sentences))

    count = 0
    count1=0

    # Counter({'REST#QUALITY': 1070, 'REST#SERVICE': 388, 'REST#GENERAL': 378, 'REST#AMBIENCE': 352, 'REST#PRICES': 342, 'REST#STYLEOPTIONS': 299, 'REST#LOCATION': 155})
    for i in root.iter('sentence'):
        # print("la: " + str(count_1))
        if (i.get('OutOfScope') != 'TRUE'):
            for opi in i.iter('Opinion'):
                name = opi.get('category')
                if (name == name_cate ):
                    text = i.find('text').text
                    datas_valid1.append(text)
                    labels_valid1.append(name)
                    count += 1
    print(count)
    for i in root.iter('sentence'):
        # print("la: " + str(count_1))
        if (i.get('OutOfScope') != 'TRUE'):
            for opi in i.iter('Opinion'):
                name = opi.get('category')
                if (name != name_cate):
                    text = i.find('text').text
                    if(text not in datas_valid1):
                        datas_valid1.append(text)
                        labels_valid1.append('None')
                        count1 += 1
    print(count1)
    labels_valid = []
    for i in labels_valid1:
        labels_valid.append(i + "\n")
    SPECIAL_CHARACTER = '%@$=+-,.:"!@#$%^&*(){}[]\/?;!;🏻/()👍*❤"😍&^:♥<>#|\n\t\''
    datas_valid = []
    for i in datas_valid1:
        my_words = i.split(" ")
        for word1 in i:
            if word1 in SPECIAL_CHARACTER:
                i = i.replace(word1, "")
                i = i.replace("  ", " ")
        for word in my_words:
            if len(word) > 20 or len(word) < 2:
                i = i.replace(word, "")
                i = i.replace("  ", " ")
        i = ViTokenizer.tokenize(i)
        my_words = i.split(" ")
        for word in my_words:
            if word in StopWords.STOP_WORDS:
                i = i.replace(word, "")
                i = i.replace("  ", " ")
        i = i.lower()
        # print(i)
        datas_valid.append(i)
    return datas_valid,labels_valid

NB=[]
SVC=[]
RF=[]

#AMBIENCE
datas_valid,labels_valid = Load_data("REST#AMBIENCE")
print(datas_valid)
f1Score_ambience = []
X_valid = datas_valid
load_file = open(join("models_NB_BOW","AMBIENCE_NB_BOW.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
print(len(a),len(labels_valid),len(datas_valid))
# print(recall_score(labels_valid,a,average='weighted'))
f1Score_ambience.append(f1_score(labels_valid, a,average='weighted')*100)
NB.append(f1_score(labels_valid,a,average='weighted')*100)

load_file = open(join("models_SVC_BOW","AMBIENCE_SVM_BOW.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
# print(recall_score(labels_valid,a,average='weighted'))
f1Score_ambience.append(f1_score(labels_valid, a,average='weighted')*100)
SVC.append(f1_score(labels_valid,a,average='weighted')*100)

load_file = open(join("models_RF_BOW","AMBIENCE_RF_BOW.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
# print(recall_score(labels_valid,a,average='weighted'))
# print(a)
# print(labels_valid)
f1Score_ambience.append(f1_score(labels_valid, a,average='weighted')*100)
RF.append(f1_score(labels_valid,a,average='weighted')*100)
print(f1Score_ambience)

#GENERAL
datas_valid,labels_valid = Load_data("REST#GENERAL")
f1Score_general = []
X_valid = datas_valid
load_file = open(join("models_NB_BOW","GENERAL_NB_BOW.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
# print(recall_score(labels_valid,a,average='weighted'))
f1Score_general.append(f1_score(labels_valid, a,average='weighted')*100)
NB.append(f1_score(labels_valid,a,average='weighted')*100)

load_file = open(join("models_SVC_BOW","GENERAL_SVM_BOW.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
# print(recall_score(labels_valid,a,average='weighted'))
f1Score_general.append(f1_score(labels_valid, a,average='weighted')*100)
SVC.append(f1_score(labels_valid,a,average='weighted')*100)

load_file = open(join("models_RF_BOW","GENERAL_RF_BOW.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
# print(recall_score(labels_valid,a,average='weighted'))
# print(a)
# print(labels_valid)
f1Score_general.append(f1_score(labels_valid, a,average='weighted')*100)
RF.append(f1_score(labels_valid,a,average='weighted')*100)
print(f1Score_general)

#LOCATION
datas_valid,labels_valid = Load_data("REST#LOCATION")
f1Score_location = []
X_valid = datas_valid
load_file = open(join("models_NB_BOW","LOCATION_NB_BOW.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
# print(recall_score(labels_valid,a,average='weighted'))
f1Score_location.append(f1_score(labels_valid, a,average='weighted')*100)
NB.append(f1_score(labels_valid,a,average='weighted')*100)

load_file = open(join("models_SVC_BOW","LOCATION_SVM_BOW.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
# print(recall_score(labels_valid,a,average='weighted'))
f1Score_location.append(f1_score(labels_valid, a,average='weighted')*100)
SVC.append(f1_score(labels_valid,a,average='weighted')*100)

load_file = open(join("models_RF_BOW","LOCATION_RF_BOW.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
# print(recall_score(labels_valid,a,average='weighted'))
# print(a)
# print(labels_valid)
f1Score_location.append(f1_score(labels_valid, a,average='weighted')*100)
RF.append(f1_score(labels_valid,a,average='weighted')*100)
print(f1Score_location)

#PRICES
datas_valid,labels_valid = Load_data("REST#PRICES")
f1Score_prices = []
X_valid = datas_valid
load_file = open(join("models_NB_BOW","PRICES_NB_BOW.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
# print(recall_score(labels_valid,a,average='weighted'))
f1Score_prices.append(f1_score(labels_valid, a,average='weighted')*100)
NB.append(f1_score(labels_valid,a,average='weighted')*100)

load_file = open(join("models_SVC_BOW","PRICES_SVM_BOW.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
# print(recall_score(labels_valid,a,average='weighted'))
f1Score_prices.append(f1_score(labels_valid, a,average='weighted')*100)
SVC.append(f1_score(labels_valid,a,average='weighted')*100)

load_file = open(join("models_RF_BOW","PRICES_RF_BOW.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
# print(recall_score(labels_valid,a,average='weighted'))
# print(a)
# print(labels_valid)
f1Score_prices.append(f1_score(labels_valid, a,average='weighted')*100)
RF.append(f1_score(labels_valid,a,average='weighted')*100)
print(f1Score_prices)

#QUALITY
datas_valid,labels_valid = Load_data("REST#QUALITY")
f1Score_quality = []
X_valid = datas_valid
load_file = open(join("models_NB_BOW","QUALITY_NB_BOW.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
# print(recall_score(labels_valid,a,average='weighted'))
f1Score_quality.append(f1_score(labels_valid, a,average='weighted')*100)
NB.append(f1_score(labels_valid,a,average='weighted')*100)

load_file = open(join("models_SVC_BOW","QUALITY_SVM_BOW.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
# print(recall_score(labels_valid,a,average='weighted'))
f1Score_quality.append(f1_score(labels_valid, a,average='weighted')*100)
SVC.append(f1_score(labels_valid,a,average='weighted')*100)

load_file = open(join("models_RF_BOW","QUALITY_RF_BOW.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
# print(recall_score(labels_valid,a,average='weighted'))
# print(a)
# print(labels_valid)
f1Score_quality.append(f1_score(labels_valid, a,average='weighted')*100)
RF.append(f1_score(labels_valid,a,average='weighted')*100)
print(f1Score_quality)

#SERVICE
datas_valid,labels_valid = Load_data("REST#SERVICE")
f1Score_service = []
X_valid = datas_valid
load_file = open(join("models_NB_BOW","SERVICE_NB_BOW.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
# print(recall_score(labels_valid,a,average='weighted'))
f1Score_service.append(f1_score(labels_valid, a,average='weighted')*100)
NB.append(f1_score(labels_valid,a,average='weighted')*100)

load_file = open(join("models_SVC_BOW","SERVICE_SVM_BOW.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
# print(recall_score(labels_valid,a,average='weighted'))
f1Score_service.append(f1_score(labels_valid, a,average='weighted')*100)
SVC.append(f1_score(labels_valid,a,average='weighted')*100)

load_file = open(join("models_RF_BOW","SERVICE_RF_BOW.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
# print(recall_score(labels_valid,a,average='weighted'))
# print(a)
# print(labels_valid)
f1Score_service.append(f1_score(labels_valid, a,average='weighted')*100)
RF.append(f1_score(labels_valid,a,average='weighted')*100)
print(f1Score_service)

#STYLEOPTIONS
datas_valid,labels_valid = Load_data("REST#STYLEOPTIONS")
f1Score_styleoptions = []
X_valid = datas_valid
load_file = open(join("models_NB_BOW","STYLEOPTIONS_NB_BOW.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
# print(recall_score(labels_valid,a,average='weighted'))
f1Score_styleoptions.append(f1_score(labels_valid, a,average='weighted')*100)
NB.append(f1_score(labels_valid,a,average='weighted')*100)

load_file = open(join("models_SVC_BOW","STYLEOPTIONS_SVM_BOW.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
# print(recall_score(labels_valid,a,average='weighted'))
f1Score_styleoptions.append(f1_score(labels_valid, a,average='weighted')*100)
SVC.append(f1_score(labels_valid,a,average='weighted')*100)

load_file = open(join("models_RF_BOW","STYLEOPTIONS_RF_BOW.pkl"),'rb')
clf = pickle.load(load_file)
a = clf.predict(X_valid)
# print(recall_score(labels_valid,a,average='weighted'))
# print(a)
# print(labels_valid)
# f1Score_styleoptions.append(f1_score(labels_valid, a,average='weighted')*100)
RF.append(f1_score(labels_valid,a,average='weighted')*100)
print(f1Score_styleoptions)

print(NB)
print(SVC)
print(RF)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

name =['AMBIENCE','GENERAL','LOCATION','PRICES','QUALITY','SERVICE','STYLEOPTIONS']
pd.set_option("display.max_rows",101)
df = pd.DataFrame({'models':name,'Naive_Bayes':NB,'Support_Vector_Machine': SVC,'Random_Forest': RF})
print(df)

# Setting the positions and width for the bars
pos = list(range(len(df['Naive_Bayes'])))
width = 0.25

# Plotting the bars
fig, ax = plt.subplots(figsize=(10, 5))

# Create a bar with pre_score data,
# in position pos,
plt.bar(pos,
        # using df['pre_score'] data,
        df['Naive_Bayes'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='r',
        # with label the first value in first_name
        label=df['models'][0])

# Create a bar with mid_score data,
# in position pos + some width buffer,
plt.bar([p + width for p in pos],
        # using df['mid_score'] data,
        df['Support_Vector_Machine'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='g',
        # with label the second value in first_name
        label=df['models'][1])

# Create a bar with post_score data,
# in position pos + some width buffer,
plt.bar([p + width * 2 for p in pos],
        # using df['post_score'] data,
        df['Random_Forest'],
        # of width
        width,
        # with alpha 0.5
        alpha=0.5,
        # with color
        color='b',
        # with label the third value in first_name
        label=df['models'][2])

# Set the y axis label
ax.set_ylabel('F1-Score')

# Set the chart's title
ax.set_title('Result')

# Set the position of the x ticks
ax.set_xticks([p + 1.5 * width for p in pos])

# Set the labels for the x ticks
ax.set_xticklabels(df['models'])

# Setting the x-axis and y-axis limits
plt.xlim(min(pos) - width, max(pos) + width * 4)
plt.ylim([0, 130])

# Adding the legend and showing the plot
plt.legend(['Naive Bayes', 'Support Vector Machine', 'Random Forest'], loc='upper right')
plt.grid()
plt.savefig(join("images", "result_BOW.png"))
plt.show()

df.to_csv('bow.csv')