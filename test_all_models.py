import pickle
from os.path import join
from envs.ScrapyEnviroment.Lib import os
import PreProcessing_valid

def Classification():
    # s = "Đồ ăn tại quán ăn rất là đầy đặn,đậm đà,ngon, không gian quán đẹp"
    # s= "Mình thấy suất XL ở đây to hơn, ngon hơn và rất đẹp"
    # s="Đường vào quán vẫn khó tìm , lòng_vòng lắm ."
    s="Uống soda tại quán dc tặng cả chai rất dễ_thương , salad trứng với thịt xông khói ngon cực_kì nhưng giá hơi cao , tuy_vậy vẫn sẽ quay lại quán "
    s = PreProcessing_valid.PreProcessing(s)
    print(s)
    pre = []
    pre.append(s)
    list_file = os.listdir("models_SVC_new")
    # print(list_file)
    for i in list_file:
        load_file = open(join("models_SVC_new",i),'rb')
        clf = pickle.load(load_file)
        t = clf.predict(pre)
        print(t)

if __name__ == "__main__":
    Classification()

