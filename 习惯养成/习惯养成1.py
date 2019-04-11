#!/usr/bin/env python
# coding: utf-8

# # 1 背景
# 通过分析某公司旗下的智慧乐园app提供的某小学的习惯养成数据，分析不同习惯养成的相关关系，以及习惯养成随时间的影响。

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
sns.set_style({'font.sans-serif':['simhei','Arial']})
import pylab
pylab.rcParams['figure.figsize'] = (15.0, 10.0)
df = pd.read_excel('C:/Users/Feiyang/Desktop/养习惯基础数据(西大街)(1).xlsx')
df_1 = pd.read_excel('C:/Users/Feiyang/Desktop/养习惯基础数据(西大街)(1).xlsx')
# # 2 数据初探

#发现有些数据的名字缺失，故删除这些数据
df=df.dropna()#去除缺失值


# 检查缺失值情况
df.info()

df.describe()

df.describe(include=['O'])


# # 3 数据可视化分析

# ### 3.1 id

# ### 3.2 学校
#学校占比
schools = list(df['学校'])
sch_dict = {x:schools.count(x) for x in set(schools)}
sch_df = pd.DataFrame.from_dict(sch_dict,orient='index')
sch_df.columns = ['count']
sch_df.sort_values(['count'],axis=0,ascending=False,inplace=True)
sch_df['pert'] = sch_df['count']/364862
sch_df[:20]

# ### 3.3 年级、班级、姓名
#年级占比
grades = list(df['年级'])
gra_dict = {x:grades.count(x) for x in set(grades)}
gra_df = pd.DataFrame.from_dict(gra_dict,orient='index')
gra_df.columns = ['count']
gra_df.sort_values(['count'],axis=0,ascending=False,inplace=True)
gra_df['pert'] = gra_df['count']/364862
gra_df[:20]

#班级占比
classes = list(df['班级'])
cla_dict = {x:classes.count(x) for x in set(classes)}
cla_df = pd.DataFrame.from_dict(cla_dict,orient='index')
cla_df.columns = ['count']
cla_df.sort_values(['count'],axis=0,ascending=False,inplace=True)
cla_df['pert'] = cla_df['count']/364862
cla_df

#故去重
df1=df.loc[:,['年级','班级','姓名']]
print('未去重: ', df1.shape)
print('去重: ', df1.drop_duplicates().shape)
df1=df1.drop_duplicates()

#年级与人数关系
ground_count=pd.DataFrame(df1.groupby("年级")['姓名'].count())
ground_count.columns=['count']
ground_count
ground_count.plot(kind = 'bar', stacked =True,title='年级与人数关系',figsize=(6,6))

#班级与人数关系
class_count=pd.DataFrame(df1.groupby("班级")['姓名'].count())
class_count.columns=['count']
class_count.T
class_count.plot(kind = 'bar', stacked =True,title='班级与人数关系',figsize=(10,10))


# ### 3.4学期

df['学期'].unique()

# ### 3.5 得分、完成状态、星级


df2=df.loc[:,['得分','完成状态','星级']]
df2.得分.unique()

df2.完成状态.unique()
df2.loc[df2['完成状态']=='主动完成','完成状态']=2
df2.loc[df2['完成状态']=='监督完成','完成状态']=1
df2.loc[df2['完成状态']=='未完成','完成状态']=0
df2.完成状态.unique()

df2.星级.unique()

#得分、完成状态、星级——相关关系图
correlation = df2.corr(method='spearman')
correlation 
plt.figure(figsize=(4, 4))
#绘制热力图
sns.heatmap(correlation, linewidths=0.2, vmax=1, vmin=-1, linecolor='w',
            annot=True,annot_kws={'size':8},square=True)

#完成状态占比
completion = list(df['完成状态'])
com_dict = {x:completion.count(x) for x in set(completion)}
com_df = pd.DataFrame.from_dict(com_dict,orient='index')
com_df.columns = ['count']
com_df.sort_values(['count'],axis=0,ascending=False,inplace=True)
com_df['pert'] = com_df['count']/364862
com_df

#习惯得分
habits = list(df['习惯名称'])
habits_av_score = {x:df[df.习惯名称==x]['得分'].mean() for x in set(habits)}
habits_score_df = pd.DataFrame.from_dict(habits_av_score,orient='index')
habits_score_df.columns = ['av_score']
habits_score_df.sort_values(['av_score'],axis=0,ascending=False,inplace=True)
habits_score_df

#年级得分
grades_av_score = {x:df[df.年级==x]['得分'].mean() for x in set(grades)}
grades_score_df = pd.DataFrame.from_dict(grades_av_score,orient='index')
grades_score_df.columns = ['av_score']
grades_score_df.sort_values(['av_score'],axis=0,ascending=False,inplace=True)
grades_score_df

#班级得分
classes_av_score = {x:df[df.班级==x]['得分'].mean() for x in set(classes)}
classes_score_df = pd.DataFrame.from_dict(classes_av_score,orient='index')
classes_score_df.columns = ['av_score']
classes_score_df.sort_values(['av_score'],axis=0,ascending=False,inplace=True)
classes_score_df

df= df.drop(['id','学校','学期','得分','星级','完成时间'],axis=1)




df.head()


# ### 3.5 习惯名称

#习惯占比
names=list(df['姓名'])
habits = list(df['习惯名称'])
hab_dict = {x:habits.count(x) for x in set(habits)}
hab_df = pd.DataFrame.from_dict(hab_dict,orient='index')
hab_df.columns = ['count']
hab_df.sort_values(['count'],axis=0,ascending=False,inplace=True)
hab_df['pert'] = hab_df['count']/364862
hab_df


habit_status=pd.crosstab(df['习惯名称'],df['完成状态'])
habit_status

#生成列联表 十个习惯和完成状态的关系
plt.figure(figsize=(15,10))
a=habit_status.index
status_len  = len(habit_status.count())
habit_index = np.arange(len(habit_status.index))
single_width = 0.15
for i in range(status_len):
    statusName = habit_status.columns[i]
    habitCount = habit_status[statusName]
    habitLocation = habit_index * 1.05 + (i - 1/2)*single_width
   #绘制柱形图
    plt.bar(habitLocation, habitCount, width = single_width)
    for x, y in zip(habitLocation, habitCount):
        #添加数据标签
        plt.text(x, y, '%.0f'%y, ha='center', va='bottom')
index = habit_index * 1.05 
plt.legend(loc='best')
plt.xticks(index, habit_status.index, rotation=360)
plt.title('十个习惯和完成状态的关系')

#年级、习惯与人数关系图
X = df.loc[:,['年级','习惯名称','姓名','完成状态']]
X["个数"]=1
X.drop('姓名',axis=1,inplace= True)
grade_habit=X.grouped=X.groupby(["年级","习惯名称",'完成状态'])["个数"].sum().unstack()
grade_habit.T
grade_habit.plot.bar(title='年级、习惯与人数关系')

#十个习惯和完成状态折线图
df2=X.drop('年级',axis=1)
df2.loc[df2['完成状态']=='主动完成','完成状态']=2
df2.loc[df2['完成状态']=='监督完成','完成状态']=1
df2.loc[df2['完成状态']=='未完成','完成状态']=0
habit_count=df2.groupby(['完成状态',"习惯名称"]).sum()
habit_count.unstack().plot(title='十个习惯和完成状态折线图')
print(habit_count)
sns.pairplot(habit_count.unstack(), diag_kind='kde', plot_kws={'alpha':1})

#完成状态与十种习惯绝对数以及相对数关系
a4=habit_status
a4["all"]=a4.iloc[:,0]+a4.iloc[:,1]+a4.iloc[:,2]
a4["主动完成_part"]=a4.iloc[:,0]/a4.iloc[:,3]
a4["未完成_part"]=a4.iloc[:,1]/a4.iloc[:,3]
a4["监督完成_part"]=a4.iloc[:,2]/a4.iloc[:,3]
print(a4)


#十种习惯的完成状态随时间发展的动态绝对数关系图
Y = df.loc[:,['日期','习惯名称','姓名','完成状态']]
Y["个数"]=1
Y.drop('姓名',axis=1,inplace= True)
date_habit=Y.groupby(["日期","习惯名称","完成状态"]).sum()
date_habit_count=date_habit.unstack().fillna(0)
date_habit_count=date_habit_count.unstack()
habit=['不玩手机和IPAD',"亲子共读","做小家务","写字练习"
       ,"复习后完成作业","按时起床洗漱","整理书包书桌","自己吃饭不挑食","运动健身","问候晚安"]
for i in range(10):
    pylab.rcParams['figure.figsize'] = (15.0, 100.0)
    fig = plt.figure()
    ax = fig.add_subplot(10, 1, 1)
    habitx=habit[i]
    ax.set_title(habitx+'随时间的变化')
    ax.set_xlabel('日期')
    ax.plot(date_habit_count.iloc[:,i],label="主动完成")
    ax.plot(date_habit_count.iloc[:,10+i],label="未完成")
    ax.plot(date_habit_count.iloc[:,20+i],label="监督动完成")
    for label in ax.get_xticklabels():
        label.set_visible(False)
    for label in ax.get_xticklabels()[::20]:
        label.set_visible(True)
    ax.legend(loc='best')
    
#通过图片可以发现十月到十一月之后使用app总体人数变少,数据没有可比性，故使用百分比的相对数消除样本变化带来的影响

#十种习惯的完成状态随时间发展的动态绝对数关系图
c2=date_habit.unstack().fillna(0)
c2["all"]=c2.iloc[:,0]+c2.iloc[:,1]+c2.iloc[:,2]
c2["主动完成_part"]=c2.iloc[:,0]/c2.iloc[:,3]
c2["未完成_part"]=c2.iloc[:,1]/c2.iloc[:,3]
c2["监督完成_part"]=c2.iloc[:,2]/c2.iloc[:,3]
date_habit_part=c2.drop(["个数","all"],axis=1)
date_habit_part=date_habit_part.unstack()
print(date_habit_part)
habit=['不玩手机和IPAD',"亲子共读","做小家务","写字练习"
       ,"复习后完成作业","按时起床洗漱","整理书包书桌","自己吃饭不挑食","运动健身","问候晚安"]
for i in range(10):
    pylab.rcParams['figure.figsize'] = (15.0, 100.0)
    fig = plt.figure()
    ax = fig.add_subplot(10, 1, 1)
    habitx=habit[i]
    ax.set_title(habitx)
    ax.set_xlabel('日期')
    ax.plot(date_habit_part.iloc[:,i],label="主动完成")
    ax.plot(date_habit_part.iloc[:,10+i],label="未完成")
    ax.plot(date_habit_part.iloc[:,20+i],label="监督动完成")
    for label in ax.get_xticklabels():
        label.set_visible(False)
    for label in ax.get_xticklabels()[::20]:
        label.set_visible(True)
    ax.legend(loc='best')
#可以发现十种习惯在前期主动完成会有所波动，但对着时间的发展，主动完成的百分比在增加，未完成与监督完成的百分比趋于零

#每个孩子的综合得分
name=[]
for i in names:
    if i in name:
        continue
    else:
        name.append(i)
for i in name:
    try:
        df2=df_1[df_1.姓名==i]
        score=sum(df2.得分)/len(df2)
        print(i,score)
    except:
        continue