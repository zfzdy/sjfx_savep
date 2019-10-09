# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:11:05 2019

@author: Administrator
"""

import matplotlib.pyplot as plt
import pandas as pd
import pymysql
from pandas import DataFrame
from sklearn.cluster import KMeans


#建立数据库连接
conn = pymysql.connect("130.120.3.158","dataanalysis","dataanalysis","data_analysis",charset="utf8")
print("连接成功")
#读取数据库表数据
data = pd.read_sql("SELECT * FROM yinhuan",con=conn)
d = DataFrame(data)
d.head(11)


#手肘法+聚类
'利用SSE选择k'
SSE = [] 
# 存放每次结果的误差平方和
for k in range(1,9):  
estimator = KMeans(n_clusters=k)
# 构造聚类器    
estimator.fit(d)    
SSE.append(estimator.inertia_)
X = range(1,9)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(X,SSE,'o-')
plt.show()
#K-MEANS聚类
mod = KMeans(n_clusters=max_k, max_iter = 500)#聚成max_k类数据,最大循环次数为500
r=mod.fit_predict(dataSet)+1#y_pred表示聚类的结果
#print(r)
#聚成max_k类数据，统计每个聚类下的数据量，并且求出他们的中心
r1 = pd.Series(mod.labels_).value_counts()
r2 = pd.DataFrame(mod.cluster_centers_)
r = pd.concat([r2, r1], axis = 1)
r.columns = list(dataSet.columns) + [u'类别数目']
#给每一条数据标注上被分为哪一类
r = pd.concat([dataSet, pd.Series(mod.labels_+1, index = dataSet.index)], axis = 1)
r.columns = list(dataSet.columns) + [u'聚类类别']
#r=r.sort_values(by=['道路运输','聚类类别'])
print(r.head(20))




#轮廓系数法+聚类
#从数据集中加载数据
#d2=d2.sort_values(by='道路运输',ascending=False)
dataSet = dq.loc[[201804]]
#print(dataSet)


max_silhouette_coefficient = 0
max_k = 0
max_centroids = []
max_labels_ = []
numSamples = 0
for k in range(3,10):
#设定K
    clf = KMeans(n_clusters=k)
#加载数据集合
    s = clf.fit(dataSet)
#样本数量
    numSamples = len(dataSet)
#中心点
    centroids = clf.cluster_centers_
    labels_ = clf.labels_
#获取轮廓系数
    silhouette_coefficient = metrics.silhouette_score(dataSet, clf.labels_,metric="euclidean",sample_size=numSamples)
    print("k:%d ==== silhouette_coefficient:%f"%(k,silhouette_coefficient))
#找到轮廓系数最大的K值，为效果最好的
    if max_silhouette_coefficient < silhouette_coefficient :
        max_silhouette_coefficient = silhouette_coefficient
        max_k = k
        max_centroids = centroids
        max_labels_ = labels_
#获取聚类效果值 
    print("k:%d ==== inertia_:%f"%(k,clf.inertia_))
    
print("max_k:%d ==== max_silhouette_coefficient:%f"%(max_k,max_silhouette_coefficient))    
#根据K=max_k进行K-Means聚类分析
mod = KMeans(n_clusters=max_k, max_iter = 500)#聚成max_k类数据,最大循环次数为500
r=mod.fit_predict(dataSet)+1#y_pred表示聚类的结果
#print(r)
#聚成max_k类数据，统计每个聚类下的数据量，并且求出他们的中心
r1 = pd.Series(mod.labels_).value_counts()
r2 = pd.DataFrame(mod.cluster_centers_)
r = pd.concat([r2, r1], axis = 1)
r.columns = list(dataSet.columns) + [u'类别数目']
#给每一条数据标注上被分为哪一类
r = pd.concat([dataSet, pd.Series(mod.labels_+1, index = dataSet.index)], axis = 1)
r.columns = list(dataSet.columns) + [u'聚类类别']
#r=r.sort_values(by=['道路运输','聚类类别'])
print(r.head(20))
#导入数据库
engine=create_engine('mysql+pymysql://dataanalysis(用户名):dataanalysis(密码)@130.120.3.154:3303（服务器）/data_analysis?(数据库名称)charset=utf8')
r.reset_index(level=None, drop=False, inplace=True, collevel=0, col_fill='')
print(r)
r.to_sql('yuan_2',con=engine,index=False,if_exists='append')
