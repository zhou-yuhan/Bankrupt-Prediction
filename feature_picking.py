# 特征工程代码 by 张柏舟
# 尝试使用三种方法：
# (1) 调用feature_importance函数(Gini_Importance)，选取前20个最重要的feature重新训练
# (2) 使用PCA主成分分析法
# (3) 借助已有的经济学和金融学知识，对特征进行筛选
# 所有的方法均采用随机森林法训练模型，也可以采用其他的机器学习模型


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA



# 读取数据
mydata = pd.read_csv("data.csv")

# 按照Bankrupt值排序，把所有的破产的样本放到mydata的最后几行
mydata.sort_values(by='Bankrupt?',inplace=True)
bankrupted = mydata.loc[mydata['Bankrupt?'] == 1]


# 去掉mydata前5000行（排过序所以这5000行的样本都是没破产的样本），从而破产和没破产的样本个数更平均
# 也可以尝试去掉x行，观察最终的模型性能怎么随x变化
mydata.drop(mydata.head(5000).index, inplace = True)



# 准备数据 X:features y:labels
# test_size表示测试集大小与总数据集大小之比，可以调整，现在是0.2
y = mydata["Bankrupt?"]
X = mydata.drop(columns=["Bankrupt?"], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=.2, shuffle=True)

#缩放，standarlization
scaler = StandardScaler()
scaler.fit(X)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# 训练模型的函数，输入训练集和测试集，以及使用的机器学习方法method（一个字符串）
def train_model(X_train, y_train, X_test, y_test, method):
    if method == 'LogisticRegression':
        model = LogisticRegression(max_iter=500).fit(X_train, y_train)
        print('Using Logistic Regression(max_iter = 500):')
    elif method == 'Dtree': #决策树
        model = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
        print('Using Decision Tree:')
    elif method == 'Rforest': #随机森林
        model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X_train, y_train)
        print('Using Random Forest(n_estimators=10):')
    elif method == 'Gboost': #Gradient Boost
        model = GradientBoostingClassifier().fit(X_train, y_train)
        print('Using Gradient Boost:')
    elif method == 'SVM': #支持向量机
        model = SVC().fit(X_train, y_train)
        print('Using Support Vector Machine:')
    elif method == 'MLP': #多层感知器
        model = MLPClassifier(hidden_layer_sizes=(100,100),random_state=0,max_iter=500).fit(X_train, y_train)
        print('Using Multiple-layer Perception(hidden_layer_sizes=(100,100), activation = RELU, max_iter = 500):')
    else:
        print('Not defined method')
    '''
    y_test_pred = model.predict(X_test)
    #print("Accuracy on training set: {:.4f}".format(accuracy_score(y_train, y_train_pred)))
    print("Accuracy on test set: {:.4f}".format(accuracy_score(y_test, y_test_pred)))
    #print("F1 Score on training set: {:.4f}".format(f1_score(y_train, y_train_pred)))
    precision, recall, F1, _ = precision_recall_fscore_support(y_test, y_test_pred, average="binary")
    print("Precision, Recall, F1 Score on test set: {0:.4f},{1:.4f},{2:.4f}".format(precision, recall ,F1))
    '''
    return model



# 所有的方法均以随机森林模型为例


# --------------------------------- 方法一： feature importance函数 ---------------------------------------

print("方法一：feature importance法")

# 这里我训练的是随机森林模型，可以更换为决策树或者梯度提升模型
model = train_model(X_train, y_train, X_test, y_test,'Rforest')
y_test_pred = model.predict(X_test)
before_acc = accuracy_score(y_test, y_test_pred)
before_precision, before_recall, before_F1, _ = precision_recall_fscore_support(y_test, y_test_pred, average="binary")
print("Before: accuracy={0:.4f}, precision={1:.4f}, recall={2:.4f}, f1_score={3:.4f}".format\
          (before_acc, before_precision, before_recall, before_F1))
# features是一个字符串列表，其中的每个元素是X的列名(属性名）
features = list(X.columns)

# importances是一个与features等长的列表，importance[i]为属性features[i]的重要度，大小在0-1之间
importances = model.feature_importances_ #Gini Importance

# 对importances中的元素按值从大到小排序([::-1])，indices[i]表示排序后的importances列表的importances[i]在排序前时对应的feature的索引
indices = np.argsort(importances)[::-1]

#for i in indices:
#    print ("{0}-{1:.3f}".format(features[i], importances[i]))


# 挑选前20个最重要的feature画柱状图
num_features = 20
picked_idx = indices[0:num_features]
plt.figure()
plt.title("Most Important 20 Features")
plt.bar(range(num_features), importances[picked_idx], color="g", align="center")
plt.xticks(range(num_features), [features[i] for i in picked_idx] , rotation='90')
#plt.xticks(range(num_features), picked_idx, rotation='0')
plt.xlim([-1, num_features])
plt.show()

# 重新选择数据，仅选出被选中的20列
y1 = mydata["Bankrupt?"]
X1 = mydata[[features[i] for i in picked_idx]]
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, random_state=1, test_size=.2, shuffle=True)

scaler = StandardScaler()
scaler.fit(X1)
X1_train = scaler.transform(X1_train)
X1_test = scaler.transform(X1_test)


# 重训模型
new_model1 = train_model(X1_train, y1_train, X1_test, y1_test,'Rforest')
y1_test_pred = new_model1.predict(X1_test)
after_acc = accuracy_score(y1_test, y1_test_pred)
after_precision, after_recall, after_F1, _ = precision_recall_fscore_support(y1_test, y1_test_pred, average="binary")
print("After: accuracy={0:.4f}, precision={1:.4f}, recall={2:.4f}, f1_score={3:.4f}".format\
          (after_acc, after_precision, after_recall, after_F1))

# 画柱状图比较特征工程前后的性能
before = [before_acc, before_precision, before_recall, before_F1]
after = [after_acc, after_precision, after_recall, after_F1]

labels = ["accuracy", "precision", "recall", "F1"]
fig,ax = plt.subplots(figsize=(8,5),dpi=80)
width_1 = 0.4
ax.bar(np.arange(len(before)),before,width=width_1,tick_label=labels,label = "before")
ax.bar(np.arange(len(after))+width_1,after,width=width_1,tick_label=labels,label="after")
ax.legend()
ax.set_title('Comparing Model before and after using Feature_Importance method')
plt.show()



# --------------------------------- 方法二： PCA主成分分析 ---------------------------------------

print("方法二： PCA主成分分析法")

# 特征工程之前的模型，以随机森林为例
model = train_model(X_train, y_train, X_test, y_test,'Rforest')
y_test_pred = model.predict(X_test)
before_acc = accuracy_score(y_test, y_test_pred)
before_precision, before_recall, before_F1, _ = precision_recall_fscore_support(y_test, y_test_pred, average="binary")
print("Before: accuracy={0:.4f}, precision={1:.4f}, recall={2:.4f}, f1_score={3:.4f}".format\
          (before_acc, before_precision, before_recall, before_F1))



y2 = mydata["Bankrupt?"]
X2 = mydata.drop(columns=["Bankrupt?"], axis = 1)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, random_state=2, test_size=.2, shuffle=True)

# 使用PCA对X2进行降维
pca = PCA().fit(X2)
X_train_pca = pca.transform(X2_train)
X_test_pca = pca.transform(X2_test)

scaler = StandardScaler()
scaler.fit(X_train_pca)
X_train_pca= scaler.transform(X_train_pca)
X_test_pca = scaler.transform(X_test_pca)

# 训练新的模型
new_model2 = train_model(X_train_pca, y2_train, X_test_pca, y2_test,'Rforest')
y2_test_pred = new_model2.predict(X_test_pca)
after_acc = accuracy_score(y2_test, y2_test_pred)
after_precision, after_recall, after_F1, _ = precision_recall_fscore_support(y2_test, y2_test_pred, average="binary")
print("After: accuracy={0:.4f}, precision={1:.4f}, recall={2:.4f}, f1_score={3:.4f}".format\
          (after_acc, after_precision, after_recall, after_F1))

# 画柱状图比较特征工程前后的性能
before = [before_acc, before_precision, before_recall, before_F1]
after = [after_acc, after_precision, after_recall, after_F1]

labels = ["accuracy", "precision", "recall", "F1"]
fig,ax = plt.subplots(figsize=(8,5),dpi=80)
width_1 = 0.4
ax.bar(np.arange(len(before)),before,width=width_1,tick_label=labels,label = "before")
ax.bar(np.arange(len(after))+width_1,after,width=width_1,tick_label=labels,label="after")
ax.legend()
ax.set_title('Comparing Model before and after using PCA method')
plt.show()



# --------------------------------- 方法三：基于特征本身经济学意义的特征工程 ---------------------------------------



'''
为了实施这种方法，我请教了一位光华的朋友
她在进行了一番思索后，从数据的95个特征中选取如下她认为比较重要的特征：

1：ROA(A) before interest and % after tax
   净利润与总资产的比值，描述公司赚钱的能力，即公司的每一块钱能赚多少钱
3：Operating Gross Margin/5：Operating Profit Rate
   描述公司经营性业务的盈利能力，3为毛利率、5为净利率
12:Cash Flow Rate
   现金流，长期运营下去的重要指标
13:Interest-bearing debt interest rate
   负息债务的利息,利息越高，公司越有可能破产
16：Net Value Per Share (A)
   (总资产-总负债)/股数,大概描述了公司的负债情况
18：Persistent EPS in the Last Four Seasons
   过去一年里的每股净收益,衡量公司盈利能力
31：Cash Reinvestment
   现金再投资比率，企业能用于再投资的现金是多还是少？
32: Current Ratio/33：Quick Ratio
   流动比率，流动资产/流动负债，公司会不会陷入短期的流动性问题
35/36/94: 与14类似，衡量借债的比率
41/42：Operating profit/Net profit before tax: paid-in-capital
   股东提供的每单位资金产生多少利润和税前净利润
92：ebit: earning before interest and tax
   赚的钱有多少用来付利息和税务，如果值太高意味着公司可能会破产
   
其他的feature与公司破产关系不大

'''
print("方法三：基于特征本身经济学意义的特征工程")

features = list(X.columns)
picked_idx = [1,3,12,13,16,18,31,32,35,36,94,41,42,92]
picked_features = [features[i] for i in picked_idx]

# 特征工程之前的模型，以随机森林为例
model = train_model(X_train, y_train, X_test, y_test,'Rforest')
y_test_pred = model.predict(X_test)
before_acc = accuracy_score(y_test, y_test_pred)
before_precision, before_recall, before_F1, _ = precision_recall_fscore_support(y_test, y_test_pred, average="binary")
print("Before: accuracy={0:.4f}, precision={1:.4f}, recall={2:.4f}, f1_score={3:.4f}".format\
          (before_acc, before_precision, before_recall, before_F1))


y3 = mydata["Bankrupt?"]
X3 = mydata[picked_features]
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, random_state=3, test_size=.2, shuffle=True)

scaler = StandardScaler()
scaler.fit(X3)
X3_train = scaler.transform(X3_train)
X3_test = scaler.transform(X3_test)

# 训练新的模型
new_model3 = train_model(X3_train, y3_train, X3_test, y3_test,'Rforest')
y3_test_pred = new_model3.predict(X3_test)
after_acc = accuracy_score(y3_test, y3_test_pred)
after_precision, after_recall, after_F1, _ = precision_recall_fscore_support(y3_test, y3_test_pred, average="binary")
print("After: accuracy={0:.4f}, precision={1:.4f}, recall={2:.4f}, f1_score={3:.4f}".format\
          (after_acc, after_precision, after_recall, after_F1))

# 画柱状图比较特征工程前后的性能
before = [before_acc, before_precision, before_recall, before_F1]
after = [after_acc, after_precision, after_recall, after_F1]

labels = ["accuracy", "precision", "recall", "F1"]
fig,ax = plt.subplots(figsize=(8,5),dpi=80)
width_1 = 0.4
ax.bar(np.arange(len(before)),before,width=width_1,tick_label=labels,label = "before")
ax.bar(np.arange(len(after))+width_1,after,width=width_1,tick_label=labels,label="after")
ax.legend()
ax.set_title('Comparing Model before and after using economic method')
plt.show()
