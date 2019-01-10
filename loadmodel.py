import pickle
from os.path import join
from sklearn.metrics import confusion_matrix, classification_report
import PreProcessing_valid
from lxml import etree as ET

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
    s = "Đồ ăn tại quán ăn rất là đầy đặn,đậm đà,ngon, không gian quán đẹp"
    # s= "Mình thấy suất XL ở đây to hơn, ngon hơn và rất đẹp, nhưng rất đắt"
    s = PreProcessing_valid.PreProcessing(s)
    print(s)
    pre = []
    pre.append(s)
    # datas_valid = []
    # labels_valid = []
    # vectorizer = CountVectorizer()
    # transformed_x_valid = vectorizer.fit_transform(s).toarray()
    load_file = open(join("models_SVC_new","QUALITY_new.pkl"),'rb')
    clf = pickle.load(load_file)
    # print("Loading file : ",clf)
    #
    # with open(join("data_test", "datas_GENERAL.txt"), 'r', encoding='utf-8')as file:
    #     for i in file:
    #         datas_valid.append(i)
    # with open(join("data_test", "labels_GENERAL.txt"), 'r', encoding='utf-8')as file:
    #     for i in file:
    #         labels_valid.append(i)
    print(len(datas))
    X_valid = datas
    a = clf.predict(X_valid)
    # with open("predict.txt",'w',encoding='utf-8') as f:
    #     for i in a:
    #         f.write(i)
    t = clf.predict(pre)
    print(t)
    print(confusion_matrix(categories, a))
    # print(classification_report(labels_valid,a))



if __name__ == "__main__":
    Classification()

