
import pandas as pd
from os.path import join
from lxml import etree as ET
from pyvi import ViTokenizer
import StopWords

tree = ET.parse(join("data", "rest_final_bao.xml"))
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

# Counter({'REST#QUALITY': 1070, 'REST#SERVICE': 388, 'REST#GENERAL': 378, 'REST#AMBIENCE': 352, 'REST#PRICES': 342, 'REST#STYLEOPTIONS': 299, 'REST#LOCATION': 155})
for i in root.iter('sentence'):
    # print("la: " + str(count_1))
    if (i.get('OutOfScope') != 'TRUE'):
        for opi in i.iter('Opinion'):
            name = opi.get('category')
            if (name == 'REST#STYLEOPTIONS'):
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
            if (name != 'REST#STYLEOPTIONS'):
                text = i.find('text').text
                if(text not in datas):
                    datas.append(text)
                    categories.append('None')
                    count1 += 1
print(count1)

SPECIAL_CHARACTER = '%@$=+-,.:"!@#$%^&*(){}[]\/?;!;🏻/()👍*❤"😍&^:♥<>#|\n\t\''

with open(join("data_train", "datas_STYLEOPTIONS_new.txt"),'w',encoding='utf-8') as file:
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

with open(join("data_train", "labels_STYLEOPTIONS_new.txt"),'w',encoding='utf-8') as file:
    for i in categories:
        file.write(str(i)+"\n")

print(len(datas))
print(len(categories))


