import os#s是用聚类打标签版本
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

data = pd.read_csv(r'C:\Users\fansen\Desktop/teat_result.csv')

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

X = data[['times','balance','allPay','invest']]
cloumns_fix1 = ['times','balance','allPay','invest']
y = data[['y']]
print(y)
#数据降维
# pca_2 = PCA(n_components=2)#降成几维
# data_pca_2 = pd.DataFrame(pca_2.fit_transform(X))
#

#绘制散点图查看数据点大致情况
#//数据的PCA降维度

#预计将数据点分类为2类


# print(y)
# 加载数据，查看数据
from sklearn.model_selection import train_test_split
data_train = data[0:9000]

y_train = y[0:9000]

data_test = data[9000:13727]
y_test = y[9000:13727]

X_train = data_train[['times','balance','allPay','invest']]
X_test = data_test[['times','balance','allPay','invest']]
# 使用随机森林分类器进行集成模型的训练以及预测分析。
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report
#
# rfc = RandomForestClassifier(random_state=1,n_estimators=10, min_samples_split=2, min_samples_leaf=2)
#
# rfc.fit(X_train, np.ravel(y_train))
# #
# rfc_y_pred = rfc.predict(X_test)
#
# #输出随机森林分类器在测试集上的分类准确性，以及更加详细的精确率、召回率、F1指标。
# print('The accuracy of random forest classifier is', rfc.score(X_test, y_test))
# print(classification_report(rfc_y_pred, y_test))
#XGBoost
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
weight = compute_class_weight('balanced', classes = [0,1], y = np.ravel(y))
xgb = xgb.XGBClassifier(learning_rate =0.08,n_estimators=50,max_depth=9,min_child_weight=2,gamma=0.3,subsample=0.8,colsample_bytree=1,
 objective= 'binary:logitraw',nthread=-1,scale_pos_weight=(weight[1]/weight[0]),seed=76,reg_alpha=1, reg_lambda=1)
xgb.fit(X_train,np.ravel(y_train))
y_pred = xgb.predict(X_test)
print('The accuracy of xgb is', xgb.score(X_test , y_test))
print(classification_report(y_pred, y_test))
