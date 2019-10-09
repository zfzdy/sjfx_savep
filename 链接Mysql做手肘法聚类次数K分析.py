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
