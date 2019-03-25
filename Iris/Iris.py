# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 11:18:45 2019

@author: Feiyang
"""

#https://www.cnblogs.com/star-zhao/p/9847082.html
#1. 项目背景
#鸢尾属(拉丁学名：Iris L.), 单子叶植物纲, 鸢尾科多年生草本植物, 开的花大而美丽, 观赏价值很高. 
#鸢尾属约300种, Iris数据集中包含了其中的三种: 山鸢尾(Setosa),  杂色鸢尾(Versicolour), 
#维吉尼亚鸢尾(Virginica), 每种50个数据, 共含150个数据. 在每个数据包含四个属性: 花萼长度，
#花萼宽度，花瓣长度，花瓣宽度, 可通过这四个属性预测鸢尾花卉属于 (山鸢尾, 杂色鸢尾, 维吉尼亚鸢尾) 哪一类.

#2.1 读取数据
#数据为csv文件, 读取数据
import pandas as pd
import numpy as np
pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)

df_Iris = pd.read_csv('C:/Users/Feiyang/Desktop/pycase/Iris/Iris.csv')

#2.2 查看前/后5行数据
#前5行
df_Iris.head()
#后5行
df_Iris.tail()

#2.3 查看数据整体信息
#查看数据整体信息
df_Iris.info()

#2.4 描述性统计
df_Iris.describe()
#注意这里是大写的字母O, 不是数字0.
df_Iris.describe(include =['O']).T#离散型变量的统计特征
df_Iris.Species.value_counts()

#3. 特征工程
#3.1 数据清洗
#第一种方法: 替换
#df_Iris['Species']=df_Iris['Species'].str.replace('Iris-','')
#第二种方法: 分割
df_Iris['Species']=df_Iris['Species'].apply(lambda x:x.split('-')[1])
df_Iris['Species'].unique()

#3.2 数据可视化
#3.2.1 relplot散点图
import seaborn as sns
import matplotlib.pyplot as plt
#sns初始化
sns.set()
#设置散点图x轴与y轴以及data参数
sns.relplot(x='SepalLengthCm',y='SepalWidthCm',data=df_Iris)
plt.title('SepalLengthCm and SepalWidthCm data analysize')

#hue表示按照Species对数据进行分类, 而style表示每个类别的标签系列格式不一致.
sns.relplot(x='SepalLengthCm',y='SepalWidthCm',hue='Species',style='Species',data=df_Iris)
plt.title('SepalLengthCm and SepalWidthCm data by Species')

#花瓣长度与宽度分布散点图
sns.relplot(x='PetalLengthCm', y='PetalWidthCm', hue='Species', style='Species', data=df_Iris )
plt.title('PetalLengthCm and PetalWidthCm data by Species')

#花萼与花瓣长度分布散点图
sns.relplot(x='SepalLengthCm', y='PetalLengthCm', hue='Species', style='Species', data=df_Iris )
plt.title('SepalLengthCm and PetalLengthCm data by Species')

#花萼与花瓣宽度分布散点图
sns.relplot(x='SepalWidthCm', y='PetalWidthCm', hue='Species', style='Species', data=df_Iris )
plt.title('SepalLengthCm and PetalLengthCm data by Species')

#花萼的长度与花瓣的宽度分布散点图
sns.relplot(x='SepalLengthCm', y='PetalWidthCm', hue='Species', style='Species', data=df_Iris )
plt.title('SepalLengthCm and PetalWidthCm data by Species')

#花萼的宽度与花瓣的长度分布散点图
sns.relplot(x='SepalWidthCm', y='PetalLengthCm', hue='Species', style='Species', data=df_Iris )
plt.title('SepalWidthCm and PetalLengthCm data by Species') 

#Id编号与花萼长度, 花萼宽度, 花瓣长度, 花瓣宽度之间有没有关系呢
#花萼长度与Id之间关系图
sns.relplot(x="Id", y="SepalLengthCm",hue="Species", style="Species",kind="line", data=df_Iris)
plt.title('SepalLengthCm and Id data analysize')
#花萼宽度与Id之间关系图
sns.relplot(x="Id", y="SepalWidthCm",hue="Species", style="Species",kind="line", data=df_Iris)
plt.title('SepalWidthCm and Id data analysize')
#花瓣长度与Id之间关系图
sns.relplot(x="Id", y="PetalLengthCm",hue="Species", style="Species",kind="line", data=df_Iris)
plt.title('PetalLengthCm and Id data analysize')
#花瓣宽度与Id之间关系图
sns.relplot(x="Id", y="PetalWidthCm",hue="Species", style="Species",kind="line", data=df_Iris)
plt.title('PetalWidthCm and Id data analysize')

#3.2.2 jointplot
#散点图和直方图同时显示, 可以直观地看出哪组频数最大, 哪组频数最小.
sns.jointplot(x='SepalLengthCm',y='SepalWidthCm',data=df_Iris)
sns.jointplot(x='PetalLengthCm', y='PetalWidthCm',  data=df_Iris )

#3.2.3 distplot
#对于频数的值, 在散点图上数点的话, 显然效率太低, 还易出错, 下面引出distplot
#绘制直方图, 其中kde=False表示不显示核函数估计图,这里为了更方便去查看频数而设置它为False.
#sns.distplot(df_Iris.SepalLengthCm,hist=True,bins=8,kde=False)
sns.distplot(df_Iris.SepalWidthCm,bins=13, hist=True, kde=True)
#sns.distplot(df_Iris.PetalLengthCm, bins=5, hist=False, kde=False)
#sns.distplot(df_Iris.PetalWidthCm, bins=5, hist=False, kde=True)

#3.2.4 boxplot
#boxplot所绘制的就是箱线图, 它能显示出一组数据的最大值, 最小值, 四分位数以及异常点.

#比如数据中的SepalLengthCm属性
sns.boxplot(x='SepalLengthCm',data=df_Iris)
#比如数据中的SepalWidthCm属性
sns.boxplot(x='SepalWidthCm', data=df_Iris)

#对于每个属性的data创建一个新的DataFrame
Iris1 = pd.DataFrame({'Id':np.arange(1,151),'Attribute': 'SepalLengthCm', 'Data':df_Iris.SepalLengthCm, 'Species':df_Iris.Species})
Iris2 = pd.DataFrame({"Id": np.arange(151,301), 'Attribute': 'SepalWidthCm', 'Data':df_Iris.SepalWidthCm, 'Species':df_Iris.Species})
Iris3 = pd.DataFrame({"Id": np.arange(301,451), 'Attribute': 'PetalLengthCm', 'Data':df_Iris.PetalLengthCm, 'Species':df_Iris.Species})
Iris4 = pd.DataFrame({"Id": np.arange(451,601), 'Attribute': 'PetalWidthCm', 'Data':df_Iris.PetalWidthCm, 'Species':df_Iris.Species})
#将四个DataFrame合并为一个
Iris=pd.concat([Iris1,Iris2,Iris3,Iris4])
#绘制箱线图
sns.boxplot(x='Attribute', y='Data', data=Iris)
sns.boxplot(x='Attribute', y='Data',hue='Species',data=Iris)

#3.2.5 violinplot
#violinplot绘制的是琴图, 是箱线图与核密度图的结合体, 既可以展示四分位数, 又可以展示任意位置的密度
sns.violinplot(x='Attribute', y='Data',hue='Species',data=Iris)

#花萼长度
# sns.boxplot(x='Species', y='SepalLengthCm', data=df_Iris)
# sns.violinplot(x='Species', y='SepalLengthCm', data=df_Iris)
# plt.title('SepalLengthCm data by Species')
#花萼宽度
# sns.boxplot(x='Species', y='SepalWidthCm', data=df_Iris)
# sns.violinplot(x='Species', y='SepalWidthCm', data=df_Iris)
# plt.title('SepalWidthCm data by Species')
#花瓣长度
# sns.boxplot(x='Species', y='PetalLengthCm', data=df_Iris)
# sns.violinplot(x='Species', y='PetalLengthCm', data=df_Iris)
# plt.title('PetalLengthCm data by Species')
#花瓣宽度
sns.boxplot(x='Species', y='PetalWidthCm', data=df_Iris)
sns.violinplot(x='Species', y='PetalWidthCm', data=df_Iris)
plt.title('PetalWidthCm data by Species')

#3.2.6 pairplot
#删除Id特征, 绘制分布图
sns.pairplot(df_Iris.drop('Id',axis=1),hue='Species')
#保存图片, 由于在jupyter notebook中太大, 不能一次截图
plt.savefig('pairplot.png')
plt.show()

#4. 构建模型
#采用决策树分类算法.
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
X = df_Iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = df_Iris['Species']
#将数据按照8:2的比例随机分为训练集, 测试集
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
#初始化决策树模型
dt=DecisionTreeClassifier()
#训练模型
dt.fit(X_train,y_train)
#用测试集评估模型的好坏
dt.score(X_test,y_test)

