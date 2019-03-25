# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 16:05:46 2019

@author: Feiyang
"""

#https://www.cnblogs.com/star-zhao/p/10100233.html
#1. 数据来源及背景
#该数据集是收集于联合循环发电厂的9568个数据点, 共包含5个特征: 
#每小时平均环境变量温度（AT），环境压力（AP），相对湿度（RH），
#排气真空（V）和净每小时电能输出（PE）, 其中电能输出PE是我们要预测的变量.

#2. 数据探索分析
import pandas as pd
#第二参数代表要读取的sheet, 0表示第一个, 1表示第二个..., pandas默认读取第一个
df = pd.read_excel('C:/Users/Feiyang/Desktop/pycase/CCPP/Folds.xlsx', 4)
#查看数据前3行/后3行
pd.set_option('display.max_rows', 6)
df
#查看数据整体信息
df.info()
#描述性统计
pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)
df.describe()
#偏态系数
for i in df.columns:
    print('{}偏态系数: '.format(i),df[i].skew())
#峰态系数
for i in df.columns:
    print('{}峰态系数: '.format(i),df[i].kurt())

#3. 相关分析
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set()
sns.pairplot(df)
#保存图片
plt.savefig('ccpp.png')
plt.show()

#3.2 相关系数 
#计算相关系数, 默认为皮尔逊相关系数
correlation = df.corr()
correlation

#3.3 热力图
#绘制热力图, 设置线宽, 最大值, 最小值, 线条白色, 显示数值, 方形
sns.heatmap(correlation,linewidths=0.2,vmax=1,vmin=-1,linecolor='w',square=True,annot=True)

#4. 回归分析
#4.1 划分数据集
from sklearn.model_selection import train_test_split
X, y = df[['AT', 'V', 'AP', 'RH']], df['PE']
#按照8:2的比例划分为训练数据集和测试数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#4.2 构建模型
from sklearn.linear_model import LinearRegression
#模型实例化
LR = LinearRegression()
#训练模型
LR.fit(X_train, y_train)
print("截距: ", LR.intercept_)
print("回归系数: ", LR.coef_ )

#4.3 模型评估
from sklearn import metrics
#分别对训练集和测试集进行预测
y_train_pred=LR.predict(X_train)
y_test_pred=LR.predict(X_test)
y_train_mae=metrics.mean_absolute_error(y_train,y_train_pred)
y_test_mae=metrics.mean_absolute_error(y_test,y_test_pred)
print('训练集MAE: ', y_train_mae)
print("测试集MAE: ", y_test_mae)

#分别计算训练集和测试集的均方误差
y_train_mse = metrics.mean_squared_error(y_train, y_train_pred)
y_test_mse = metrics.mean_squared_error(y_test, y_test_pred)
print('训练集MSE: ', y_train_mse)
print("测试集MSE: ", y_test_mse)

from math import sqrt
y_train_rmse = sqrt(y_train_mse)
y_test_rmse = sqrt(y_test_mse)
print('训练集RMSE: ', y_train_rmse)
print("测试集RMSE: ", y_test_rmse)

#分别计算训练集和测试集的多重判定系数
y_train_r2 = metrics.r2_score(y_train, y_train_pred)
y_test_r2 = metrics.r2_score(y_test, y_test_pred)
print('训练集R2: ', y_train_r2)
print("测试集R2: ", y_test_r2)

#直接用训练好的模型去评分
y_train_score = LR.score(X_train, y_train)
y_test_score = LR.score(X_test, y_test)
print('训练集score: ', y_train_score)
print("测试集score: ", y_test_score)

#4.4 模型检验
from pandas import Series
from scipy.stats import f
#将array转为series格式
y_train_pred = Series(y_train_pred,index=y_train.index)
#分别计算训练数据上的SSR和SSE
y_train_ssr = y_train_pred.apply(lambda x:(x-y_train.mean())**2).sum()
y_train_sse =y_train_pred.sub(y_train).apply(lambda x: x**2).sum()
#dn是SSR的自由度(自变量个数), df则是SSE的自由度
dn=4
df=y_train.shape[0]-dn-1
#计算F值
y_train_f =(y_train_ssr/dn)/(y_train_sse/df)
#计算p值
p=f.sf(y_train_f,dn,df)
#计算0.05显著性水平下临界值
cr_value =f.isf(0.05,dn,df)
print('训练数据集F值: ', y_train_f)
print('0.05显著性水平下临界值: ', cr_value)
print('训练数据集P值: %.20f'% p)

#2) 回归系数检验
from scipy.stats import t
import numpy as np
def get_tvalue(sse,df,matr,beta,i):
    '''计算t值'''
    mse=sse/df
    sbeta=sqrt(matr[i+1,i+1]*mse)
    t=beta/sbeta
    return t

limit=t.isf(0.025,df)
print('0.05显著性水平下的临界值: ', limit)

X_train['B']=1
X_train=X_train.reindex(columns=['B', 'AT', 'V', 'AP', 'RH'])

#转成矩阵
xm = np.mat(X_train)
#计算(X'X)的逆矩阵
xmi=np.dot(xm.T,xm).I
betas=LR.coef_
index=range(4)
for i,beta in zip(index,betas):
    tvalue=get_tvalue(y_train_sse,df,xmi,beta,i)
    pvalue=t.sf(abs(tvalue),df)*2
    print('beta{0}的t值: '.format(i+1), tvalue)
    print('beta{0}的p值: '.format(i+1), pvalue)
    
    
#5. 多重共线性
#删除共线变量
df = pd.read_excel('C:/Users/Feiyang/Desktop/pycase/CCPP/Folds.xlsx', 4)
X1  = df[['AT', 'AP', 'RH']] 
y1  = df['PE']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.2)
#训练模型, 预测以及计算均方根误差和多重判定系数
LR1 = LinearRegression()
LR1.fit(X1_train, y1_train)
inter, co = LR1.intercept_, LR1.coef_
y1_train_pred = LR1.predict(X1_train)
y1_train_rmse = sqrt(metrics.mean_squared_error(y1_train, y1_train_pred))
y1_train_score = LR1.score(X1_train, y1_train)
print("回归模型: PE={0}+{1}AT+{2}AP+{3}RH".format(inter,co[0], co[1], co[2]))
print('训练集RMSE: ', y1_train_rmse)
print('训练集拟合优度: ', y1_train_score)

#计算F检验中0.05显著性水平下的P值
y1_train_pred = Series(y1_train_pred, index=y1_train.index)
y1_train_ssr = y1_train_pred.apply(lambda x: (x - y1_train.mean())**2).sum()
y1_train_sse = y1_train.sub(y1_train_pred).apply(lambda x: x**2).sum()
dn1, df1 = 3, y1_train.shape[0]-3-1
y1_train_f = (y1_train_ssr/dn1) / (y1_train_sse/df1)
y1_p = f.sf(y1_train_f, dn1, df1)
# cr_value = f.isf(0.05, dn, df)
print('F检验 0.05显著性水平下训练数据的P值: %.20f'% y1_p)

#计算t检验在0.05显著性水平下的P值
def get_t1value(sse, df, matr, beta, i):
    mse = sse / df
    sbeta = sqrt(matr[i+1, i+1]* mse)
    t = beta / sbeta
    return t
X1_train['B'] = 1
X1_train = X1_train.reindex(columns=['B', 'AT', 'AP', 'RH'])
xm1 = np.mat(X1_train)
xmi1 = np.dot(xm1.T, xm1).I
index, betas = range(3), LR1.coef_
for i, beta in zip(index, betas):
    tvalue = get_t1value(y1_train_sse, df1, xmi1, beta, i)
    pvalue = t.sf(abs(tvalue), df1)*2
    print('t检验 0.05显著性水平下beta{0}的p值: '.format(i+1), pvalue)
    
#2. 转换模型形式: 将数据进行对数转换
df_log = np.log(df)
X2  = df[['AT','V', 'AP', 'RH']] 
y2  = df['PE']
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size = 0.2)
#训练模型, 预测以及计算均方根误差和多重判定系数
LR2 = LinearRegression()
LR2.fit(X2_train, y2_train)
inter, co = LR2.intercept_, LR2.coef_
y2_train_pred = LR2.predict(X2_train)
y2_train_rmse = sqrt(metrics.mean_squared_error(y2_train, y2_train_pred))
y2_train_score = LR2.score(X2_train, y2_train)
print("回归模型: PE={0}+{1}AT+{2}V+{3}AP+{4}RH".format(inter,co[0], co[1], co[2],co[3]))
print('训练集RMSE: ', y2_train_rmse)
print('训练集拟合优度: ', y2_train_score)

#计算F检验中0.05显著性水平下的P值
y2_train_pred = Series(y2_train_pred, index=y2_train.index)
y2_train_ssr = y2_train_pred.apply(lambda x: (x - y2_train.mean())**2).sum()
y2_train_sse = y2_train.sub(y2_train_pred).apply(lambda x: x**2).sum()
dn2, df2 = 4, y2_train.shape[0]-4-1
y2_train_f = (y2_train_ssr/dn1) / (y2_train_sse/df1)
y2_p = f.sf(y2_train_f, dn1, df1)
# cr_value = f.isf(0.05, dn, df)
print('F检验 0.05显著性水平下训练数据的P值: %.20f'% y2_p)

#计算t检验在0.05显著性水平下的P值
def get_t2value(sse, df, matr, beta, i):
    mse = sse / df
    sbeta = sqrt(matr[i+1, i+1]* mse)
    t = beta / sbeta
    return t
X2_train['B'] = 1
X2_train = X2_train.reindex(columns=['B', 'AT','V', 'AP', 'RH'])
xm2 = np.mat(X2_train)
xmi2 = np.dot(xm2.T, xm2).I
index, betas = range(4), LR1.coef_
for i, beta in zip(index, betas):
    tvalue = get_t2value(y2_train_sse, df1, xmi1, beta, i)
    pvalue = t.sf(abs(tvalue), df2)*2
    print('t检验 0.05显著性水平下beta{0}的p值: '.format(i+1), pvalue)
    
# 6. 模型应用
import matplotlib.pyplot as plt
#用训练好的模型去预测
y2_test_pred = LR2.predict(X2_test)
y2_test_rmse = sqrt(metrics.mean_squared_error(y2_test, y2_test_pred))
# y1_test_rmse = sqrt(metrics.mean_squared_error(y1_test, y1_test_pred))
y2_test_score = LR2.score(X2_test, y2_test)
print('测试集RMSE: ', y2_test_rmse)
print('测试集拟合优度: ', y2_test_score)
#绘制曲线
plt.figure(figsize=(12,8))
plt.plot(range(len(y2_test)), y2_test, 'g', label='test data')
plt.plot(range(len(y2_test_pred)), y2_test_pred, 'r', label='predict data', linewidth=2, alpha=0.8)
plt.legend()
plt.savefig('tp.png')
plt.show()
    
