import pickle
from os.path import join

from pyvi import ViTokenizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score,f1_score
import PreProcessing_valid
from lxml import etree as ET

import StopWords

tree = ET.parse(join("data", "data_test_doan.xml"))
root = tree.getroot()
datas = []
categories = []
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
            if (name == 'REST#QUALITY'):
                text = i.find('text').text
                datas.append(text)
                categories.append(name)
                count += 1
print(count)
for i in root.iter('sentence'):
    # print("la: " + str(count_1))
    if (i.get('OutOfScope') != 'TRUE'):
        for opi in i.iter('Opinion'):
            name = opi.get('category')
            if (name != 'REST#QUALITY'):
                text = i.find('text').text
                if(text not in datas):
                    datas.append(text)
                    categories.append('None')
                    count1 += 1
print(count1)

SPECIAL_CHARACTER = '%@$=+-,.:"!@#$%^&*(){}[]\/?;!;🏻/()👍*❤"😍&^:♥<>#|\n\t\''
def Classification():
    datas_new = []
    for i in datas:
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
        datas_new.append(i)
    print(len(datas_new))
    load_file = open(join("models_SVC_new", "QUALITY_new.pkl"), 'rb')
    clf = pickle.load(load_file)
    c = clf.predict(datas_new)
    # print(c)
    labels = []
    for i in categories:
        labels.append(i+"\n")
    # print(labels)
    # print(c)
    print(accuracy_score(labels, c))
    print(confusion_matrix(labels, c))
    print(classification_report(labels, c))
    print(f1_score(labels,c,average='weighted'))
    # print(c)

if __name__ == "__main__":
    Classification()
