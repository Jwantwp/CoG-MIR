import pickle
import os




# 打开 .pkl 文件并加载内容
with open('/home/sharing/disk1/disk1/wangpeiwu/wangpeiwu/Study_1/pro_MIntRec_reason_all.pkl', 'rb') as file:
    data = pickle.load(file)
    

# 现在 data 就是原来保存的对象
# print(data.keys())
print(data['MIntRec_S05_E16_329'])
print(data['MIntRec_S05_E16_329'].shape)  #(32, 768)