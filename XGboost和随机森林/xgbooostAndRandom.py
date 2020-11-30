import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import openpyxl
from openpyxl import *
import xlwt,xlrd
import xlutils
from xlutils.copy import copy
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
path = r'C:\Users\fansen\Desktop'
excelFile = os.path.join(path,'2.xlsx')
data_xls = pd.read_excel(excelFile, index_col=0,)#index是指指定列是索引列，skip是忽略第一行
data_xls.to_csv('pktrain.csv')
print(data_xls)

data_train = pd.read_csv('pktrain.csv')
print(data_train)
# data_test = pd.read_csv('pktest.csv')#分出来一部分数据做测试集

# data_train.head(4)
# # data_test.head(4)
# # LabelEncoder编码模式
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# # 添加一个新的列
# data_train["Sex"] = le.fit_transform(data_train["性别"])
# data_test["Sex"]=le.fit_transform(data_test["性别"])
# data_train = data_train.drop(["性别"],axis=1)
# data_test = data_test.drop(["性别"],axis=1)

X = data_train[['Prin1','Prin2','Prin3','Prin4']]
cloumns_fix1 = ['Prin1','Prin2','Prin3','Prin4']
y = data_train[['y']]
print(y)
#数据降维
pca_2 = PCA(n_components=2)#降成几维
data_pca_2 = pd.DataFrame(pca_2.fit_transform(X))


#绘制散点图查看数据点大致情况
#//数据的PCA降维度

#预计将数据点分类为3类
kmmodel = KMeans(n_clusters=2) #创建模型
kmmodel = kmmodel.fit(data_train[cloumns_fix1]) #训练模型
ptarget = kmmodel.predict(data_train[cloumns_fix1]) #对原始数据进行标注

print(ptarget)
y = ptarget
print(y)
# 加载数据，查看数据
from sklearn.model_selection import train_test_split


# 使用随机森林分类器进行集成模型的训练以及预测分析。
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

rfc = RandomForestClassifier(random_state=1,n_estimators=100, min_samples_split=4, min_samples_leaf=2)

rfc.fit(X, np.ravel(y))
#
rfc_y_pred = rfc.predict(X)

# 输出随机森林分类器在测试集上的分类准确性，以及更加详细的精确率、召回率、F1指标。
print('The accuracy of random forest classifier is', rfc.score(X, y))
print(classification_report(rfc_y_pred, y))

# import xgboost as xgb
# from sklearn.metrics import accuracy_score
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import classification_report
# weight = compute_class_weight('balanced', classes = [0,1], y = np.ravel(y))
# xgb = xgb.XGBClassifier(learning_rate =0.002,n_estimators=50,max_depth=9,min_child_weight=2,gamma=0,subsample=0.6,colsample_bytree=0.8,
#  objective= 'binary:logitraw',nthread=4,scale_pos_weight=(weight[1]/weight[0]),seed=76,reg_alpha=1, reg_lambda=1)
# xgb.fit(X,y)
# y_pred = xgb.predict(X)
# print('The accuracy of xgb is', xgb.score(X, y))
# print(classification_report(y_pred, y))