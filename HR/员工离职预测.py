#!/usr/bin/env python
# coding: utf-8

# # 1. 数据来源及背景
# 数据来源: https://www.kaggle.com/jiangzuo/hr-comma-sep/version/1
# 
# 数据背景: 该数据集是指某公司员工的离职数据, 其包含14999个样本以及10个特征, 这10个特征分别为: 员工对公司满意度, 最新考核评估, 项目数, 平均每月工作时长, 工作年限, 是否出现工作事故, 是否离职, 过去5年是否升职, 岗位, 薪资水平. 

# # 2. 明确分析目的
# 将上述影响因素与现有的数据相结合来提出问题，进而明确我们的分析目的：
# 
# 1) 员工对公司满意度平均水平如何？员工的最新考核情况又是如何？员工所参加项目数是怎样？员工平均每月工作时长以及平均工作年限分别是多少？
# 
# 2) 当前离职率是多少？工作事故发生率？过去5年升职率？薪资水平又如何？共有多少种岗位？
# 
# 3) 是否离职和其他9个特征的关系如何？
# 
# 4) 根据现有数据, 如何对某个员工是否离职进行预测？
# 
# 5) 针对当前的员工离职情况，企业该如何对待呢？

# # 3. 数据探索分析

# ### 1) 查看前2行和后2行数据

# In[1]:


import pandas as pd
df=pd.read_csv("C:/Users/Feiyang/Desktop/pycase/HR/HR_comma_sep.csv")
pd.set_option("display.max_row",4)
df


# ### 2) 查看数据类型等信息

# In[2]:


df.info()


# ### 3). 描述性统计

# In[3]:


df.describe()


# In[4]:


df.describe(include="O").T


# # 4. 数据预处理

# ### 1. 异常值

# In[5]:


import seaborn as sns
import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,5,figsize=(12,2))
sns.boxplot(x=df.columns[0],data=df,ax=ax[0])
sns.boxplot(x=df.columns[1],data=df,ax=ax[1])
sns.boxplot(x=df.columns[2],data=df,ax=ax[2])
sns.boxplot(x=df.columns[3],data=df,ax=ax[3])
sns.boxplot(x=df.columns[4],data=df,ax=ax[4])


# # 5. 可视化分析

# ### 1. 人力资源总体情况

# In[39]:


from pyecharts import Pie


# In[40]:


from pyecharts import Pie
attr = ["离职", "在职"]
v1 =[df.left.value_counts()[1], df.left.value_counts()[0]]
pie = Pie("该公司人力资源总体情况", title_pos='center')
pie.add(
    "图1",
    attr,                       
    v1,                          
    radius=[35, 65],
    label_text_color=None,
    is_label_show=True,
    legend_orient="vertical",
    legend_pos="left"
)
pie.render()


# ### 2. 对公司满意度与是否离职的关系

# In[ ]:


from pyecharts import Boxplot
#字段重命名
df.columns=['satisfaction', 'evaluation', 'project', 'hours', 'years_work','work_accident', 'left', 'promotion', 'department', 'salary']
#绘制箱线图
boxplot = Boxplot("对公司满意度与是否离职关系图", title_pos='center')
x_axis = ['在职', '离职']
y_axis = [df[df.left == 0].satisfaction.values, df[df.left == 1].satisfaction.values]
boxplot.add("", x_axis, boxplot.prepare_data(y_axis))
boxplot.render()


# ### 3. 最新考核评估与是否离职的关系

# In[ ]:


boxplot = Boxplot("最新评估与是否离职关系图", title_pos='center')
x_axis = ['在职', '离职']
y_axis = [df[df.left == 0].evaluation.values, df[df.left == 1].evaluation.values]
boxplot.add("", x_axis, boxplot.prepare_data(y_axis))
boxplot.render()


#  ### 4. 所参加项目与是否离职的关系

# In[ ]:


from pyecharts import Bar, Pie, Grid
#按照项目数分组分别求离职人数和所有人数
project_left_1 = df[df.left == 1].groupby('project')['left'].count()
project_all = df.groupby('project')['left'].count()
#分别计算离职人数和在职人数所占比例
project_left1_rate = project_left_1 / project_all
project_left0_rate = 1 - project_left1_rate
attr = project_left1_rate.index
bar = Bar("所参加项目数与是否离职的关系图", title_pos='10%')
bar.add("离职", attr, project_left1_rate, is_stack=True)
bar.add("在职", attr, project_left0_rate, is_stack=True, legend_pos="left", legend_orient="vertical")
#绘制圆环图
pie = Pie("各项目数所占百分比", title_pos='center')
pie.add('', project_all.index, project_all, radius=[35, 60], label_text_color=None, 
        is_label_show=True, legend_orient="vertical", legend_pos="67%")
grid = Grid(width=1200)
grid.add(bar, grid_right="67%")
grid.add(pie)
grid.render()


# ### 5. 平均每月工作时长和是否离职的关系

# In[ ]:


boxplot = Boxplot("平均每月工作时长与是否离职关系图", title_pos='center')
x_axis = ['在职', '离职']
y_axis = [df[df.left == 0].hours.values, df[df.left == 1].hours.values]
boxplot.add("", x_axis, boxplot.prepare_data(y_axis))
boxplot.render()


# ### 6. 工作年限和是否离职的关系

# In[ ]:


from pyecharts import Bar, Pie, Grid
#按照工作年限分别求离职人数和所有人数
years_left_0 = df[df.left == 0].groupby('years_work')['left'].count()
years_all = df.groupby('years_work')['left'].count()
#分别计算离职人数和在职人数所占比例
years_left0_rate = years_left_0 / years_all
years_left1_rate = 1 - years_left0_rate
attr = years_all.index
bar = Bar("工作年限与是否离职的关系图", title_pos='10%')
bar.add("离职", attr, years_left1_rate, is_stack=True)
bar.add("在职", attr, years_left0_rate, is_stack=True, legend_pos="left" , legend_orient="vertical")
#绘制圆环图
pie = Pie("各工作年限所占百分比", title_pos='center')
pie.add('', years_all.index, years_all, radius=[35, 60], label_text_color=None, 
        is_label_show=True, legend_orient="vertical", legend_pos="67%")
grid = Grid(width=1200)
grid.add(bar, grid_right="67%")
grid.add(pie)
grid.render()


# ### 7. 是否发生工作事故与是否离职的关系

# In[ ]:


from pyecharts import Bar
accident_left = pd.crosstab(df.work_accident, df.left)
attr = accident_left.index
bar = Bar("是否发生工作事故与是否离职的关系图", title_pos='center')
bar.add("离职", attr, accident_left[1], is_stack=True)
bar.add("在职", attr, accident_left[0], is_stack=True, legend_pos="left" , legend_orient="vertical", is_label_show=True)
bar.render()


# ### 8. 5年内是否升职与是否离职的关系

# In[ ]:


promotion_left = pd.crosstab(df.promotion, df.left)
attr = promotion_left.index
bar = Bar("5年内是否升职与是否离职的关系图", title_pos='center')
bar.add("离职", attr, promotion_left[1], is_stack=True)
bar.add("在职", attr, promotion_left[0], is_stack=True, legend_pos="left" , legend_orient="vertical", is_label_show=True)
bar.render()


# ### 9. 岗位与是否离职的关系

# In[ ]:


#分别计算各岗位离职人员比例和各岗位占总体百分比
department_left_0 = df[df.left == 0].groupby('department')['left'].count()
department_all = df.groupby('department')['left'].count()
department_left0_rate = department_left_0 / department_all
department_left1_rate = 1 - department_left0_rate
attr = department_all.index
bar = Bar("岗位与离职比例的关系图", title_top='40%')
bar.add("离职", attr, department_left1_rate, is_stack=True)
bar.add("在职", attr, department_left0_rate, is_stack=True, is_datazoom_show=True,
        xaxis_interval=0, xaxis_rotate=30,  legend_top="45%",  legend_pos="80%")
#绘制圆环图
pie = Pie("各个岗位所占百分比", title_pos='left')
pie.add('', department_all.index, department_all,center=[50, 23], radius=[18, 35], label_text_color=None, 
        is_label_show=True, legend_orient="vertical", legend_pos="80%", legend_top="4%")
grid = Grid(width=1200, height=700)
grid.add(bar, grid_top="50%", grid_bottom="25%")
grid.add(pie)
grid.render()


# ### 10. 薪资水平和是否离职的关系

# In[ ]:


from pyecharts import Bar
#按照薪资水平分别求离职人数和所有人数
salary_left = pd.crosstab(df.salary, df.left).sort_values(0, ascending = False)
attr = salary_left.index
bar = Bar("薪资水平和是否离职的关系图", title_pos='center')
bar.add("离职", attr, salary_left[1], is_stack=True)
bar.add("在职", attr, salary_left[0], is_stack=True, legend_pos="left" , legend_orient="vertical", is_label_show=True)
bar.render()


# # 6. 特征工程

# ### 1. 离散型数据处理

# #### 1) 定序

# In[44]:


df['salary']=df['salary'].map({"low": 0, "medium": 1, "high": 2})
df.salary.unique()


# #### 2) 定类

# In[48]:


#岗位是定类型变量, 对其进行one-hot编码, 这里直接利用pandas的get_dummies方法
df_one_hot = pd.get_dummies(df, prefix="dep")
df_one_hot.head()


# ### 2. 连续型数据处理

# In[ ]:


#采用max-min归一化方法
hours = df_one_hot['hours']
df_one_hot['hours'] = df_one_hot.hours.apply(lambda x: (x-hours.min()) / (hours.max()-hours.min()))


# ### 3. 相关系数

# In[ ]:


#计算相关系数
correlation = df_one_hot.corr(method = "spearman")
plt.figure(figsize=(18, 10))
#绘制热力图
sns.heatmap(correlation, linewidths=0.2, vmax=1, vmin=-1, linecolor='w',fmt='.2f',
            annot=True,annot_kws={'size':10},square=True)


# # 7. 逻辑回归模型

# ### 1. 划分数据集

# In[ ]:


from sklearn.model_selection import train_test_split
#划分训练集和测试集
X = df_one_hot.drop(['left'], axis=1)
y = df_one_hot['left']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# ### 2. 训练模型

# In[ ]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
print(LR.fit(X_train, y_train))
print("训练集准确率: ", LR.score(X_train, y_train))
print("测试集准确率: ", LR.score(X_test, y_test))


# In[ ]:


#指定随机梯度下降优化算法
LR = LogisticRegression(solver='saga')
print(LR.fit(X_train, y_train))
print("训练集准确率: ", LR.score(X_train, y_train))
print("测试集准确率: ", LR.score(X_test, y_test))


# ### 3. 调参

# In[ ]:


#用准确率进行10折交叉验证选择合适的参数C
from sklearn.linear_model import LogisticRegressionCV
Cs = 10**np.linspace(-10, 10, 400)
lr_cv = LogisticRegressionCV(Cs=Cs, cv=10, penalty='l2', solver='saga',  max_iter=10000, scoring='accuracy')
lr_cv.fit(X_train, y_train)
lr_cv.C_


# In[ ]:


LR = LogisticRegression(solver='saga', penalty='l2', C=25.52908068)
print("训练集准确率: ", LR.score(X_train, y_train))
print("测试集准确率: ", LR.score(X_test, y_test))


# ### 4. 混淆矩阵

# In[ ]:


from sklearn import metrics
X_train_pred = LR.predict(X_train)
X_test_pred = LR.predict(X_test)
print('训练集混淆矩阵:')
print(metrics.confusion_matrix(y_train, X_train_pred))
print('测试集混淆矩阵:')
print(metrics.confusion_matrix(y_test, X_test_pred))


# In[ ]:


from sklearn.metrics import classification_report
print('训练集:')
print(classification_report(y_train, X_train_pred))
print('测试集:')
print(classification_report(y_test, X_test_pred))


# # 8. 朴素贝叶斯模型

# In[ ]:


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
#构建高斯朴素贝叶斯模型
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print("训练集准确率: ", gnb.score(X_train, y_train))
print("测试集准确率: ", gnb.score(X_test, y_test))
X_train_pred =gnb.predict(X_train)
X_test_pred = gnb.predict(X_test)
print('训练集混淆矩阵:')
print(metrics.confusion_matrix(y_train, X_train_pred))
print('测试集混淆矩阵:')
print(metrics.confusion_matrix(y_test, X_test_pred))
print('训练集:')
print(classification_report(y_train, X_train_pred))
print('测试集:')
print(classification_report(y_test, X_test_pred))


# # 9. 模型评估之ROC曲线

# In[ ]:


from sklearn import metrics
from sklearn.metrics import roc_curve
#将逻辑回归模型和高斯朴素贝叶斯模型预测出的概率均与实际值通过roc_curve比较返回假正率, 真正率, 阈值
lr_fpr, lr_tpr, lr_thresholds = roc_curve(y_test, LR.predict_proba(X_test)[:,1])
gnb_fpr, gnb_tpr, gnb_thresholds = roc_curve(y_test, gnb.predict_proba(X_test)[:,1])
#分别计算这两个模型的auc的值, auc值就是roc曲线下的面积
lr_roc_auc = metrics.auc(lr_fpr, lr_tpr)
gnb_roc_auc = metrics.auc(gnb_fpr, gnb_tpr)
plt.figure(figsize=(8, 5))
plt.plot([0, 1], [0, 1],'--', color='r')
plt.plot(lr_fpr, lr_tpr, label='LogisticRegression(area = %0.2f)' % lr_roc_auc)
plt.plot(gnb_fpr, gnb_tpr, label='GaussianNB(area = %0.2f)' % gnb_roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()


# 5) 针对当前的员工离职情况，企业该如何对待呢？
# 
# 该公司的员工离职现状:
# 
# 1.当前离职率为23.81%
# 
# 2.离职人员对公司满意度普遍较低
# 
# 3.离职人员的考核成绩相对较高, 说明离职人员多数为优秀人才.
# 
# 4.项目数范围为2~7个, 其中参加7个项目的离职率最高，其次是2个的; 7个的工作能力较强, 在其他企业有更好的发展, 2个的可能是在该公司中工作能力不被认可
# 
# 5.离职人员的平均每月工作时长较长
# 
# 6.离职人员的工作年限集中在3到6年
# 
# 7.5年内未升职的离职率较高
# 
# 8.hr岗位的离职率最高, 目前企业普遍存在"留人难, 招人难”，这可能是导致该岗位的离职率高的主要原因
# 
# 9.低等薪资水平的离职率最高
# 
# 由于企业培养人才是需要大量的成本, 为了防止人才再次流失, 因此应当注重解决人才的流失问题, 也就是留人, 另外如果在招人时注意某些问题, 也能在一定程度上减少人才流失. 因此, 这里可将对策分为两种, 一种是留人对策, 一种是招人对策.
# 
# 留人对策:
# 
# 1.建立良好的薪酬制度, 不得低于市场水平
# 
# 2.建立明朗的晋升机制
# 
# 3.完善奖惩机制, 能者多劳, 也应多得.
# 
# 4.实现福利多样化, 增加员工对企业的忠诚度
# 
# 5.重视企业文化建设, 树立共同的价值观
# 
# 6.改善办公环境以及营造良好的工作氛围
# 
# 7.鼓励员工自我提升
# 
# 招人对策:
# 
# 1.明确企业招聘需求, 员工的能力应当与岗位需求相匹配
# 
# 2.与应聘者坦诚相见
# 
# 3.招聘期间给予的相关承诺必须实现
# 
# 4.欢迎优秀流失人才回归

# https://www.cnblogs.com/star-zhao/p/10186417.html
