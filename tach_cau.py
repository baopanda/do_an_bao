s = "Mình cũng ít khi uống trà sữa, chỉ là người yêu mình thích hường huệ nên cứ lôi mình vào đây..... Người yêu mình gọi hai cốc trà xanh chanh dây, trông cốc khá là đẹp, không giảm đường đá uống khá vừa. Cửa hàng đang rất đông deli đứng đợi nên bạn order có hẹn mình bị dồn đơn nên phải đợi khoảng 20p, nhưng cuối cùng mới hơn 10p đã có đồ rồi. Khá ưng. Sẽ thường xuyên dắt ng yêu qua trà sữa hồng hồng nhiều"
from nltk.tokenize import sent_tokenize

# Split text into sentences
sentences = s.split(".")
sentences_new = []
for i in sentences:
    if(i != ''):
        print(1)
        i = i.strip(' ')
        sentences_new.append(i)
print(sentences_new)

a = sent_tokenize(s)
print(a)