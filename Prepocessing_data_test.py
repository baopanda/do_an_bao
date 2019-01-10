
import pandas as pd
from os.path import join
from lxml import etree as ET
from pyvi import ViTokenizer
import StopWords

tree = ET.parse(join("data", "data_valid_new.xml"))
root = tree.getroot()
datas = []
categories = []
reviews = root.findall("Review")
sentences = root.findall("**/sentence")
print("# Reviews   : ", len(reviews))
print("# Sentences : ", len(sentences))

count = 0
count1=0
count2=0
count3=0
count4=0
count5=0
count6=0

# Counter({'REST#AMBIENCE': 47, 'REST#GENERAL': 41, 'REST#QUALITY': 38, 'REST#SERVICE': 35, 'REST#PRICES': 20, 'REST#LOCATION': 9, 'REST#STYLEOPTIONS': 7})
for i in root.iter('sentence'):
    # print("la: " + str(count_1))
    if (i.get('OutOfScope') != 'TRUE'):
        for opi in i.iter('Opinion'):
            name = opi.get('category')
            if (name == 'REST#LOCATION'):
                text = i.find('text').text
                datas.append(text)
                categories.append(name)
                count += 1
print(count)

SPECIAL_CHARACTER = '%@$=+-,.:"!@#$%^&*(){}[]\/?;!;🏻/()👍*❤"😍🤩&^:♥<>#|\n\t\''

with open(join("data_test", "datas_LOCATION.txt"),'w',encoding='utf-8') as file:
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
        file.write(i+"\n")

with open(join("data_test", "labels_LOCATION.txt"),'w',encoding='utf-8') as file:
    for i in categories:
        file.write(str(i)+"\n")

print(len(datas))
print(len(categories))


