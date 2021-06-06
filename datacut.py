# Author: 宋铭宇

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


# 读取数据
mydata = pd.read_csv("data.csv")

# 按照Bankrupt值排序，把所有的破产的样本放到mydata的最后几行
mydata.sort_values(by='Bankrupt?',inplace=True)
bankrupted = mydata.loc[mydata['Bankrupt?'] == 1]

# 准备数据 X:features y:labels
# test_size表示测试集大小与总数据集大小之比，可以调整，现在是0.2
y = mydata["Bankrupt?"]
X = mydata.drop(columns=["Bankrupt?"])
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
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    #print("Accuracy on training set: {:.4f}".format(accuracy_score(y_train, y_train_pred)))
    print("Accuracy on test set: {:.4f}".format(accuracy_score(y_test, y_test_pred)))
    #print("F1 Score on training set: {:.4f}".format(f1_score(y_train, y_train_pred)))
    print("F1 Score on test set: {:.4f}".format(f1_score(y_test, y_test_pred)))


from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN

X_resampled, y_resampled = X_train, y_train
choice = 0
if choice == 0:
    '''不做任何裁剪'''
    X_resampled, y_resampled = X_train, y_train
elif choice == 1:
    '''over-sampling,随机过采样'''
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
elif choice == 2:
    '''over-sampling,SMOTE'''
    X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)
elif choice == 3:
    '''over-sampling,ADASYN'''
    X_resampled, y_resampled = ADASYN().fit_resample(X_train, y_train)
elif choice == 4:
    '''over-sampling,BorderlineSMOTE'''
    X_resampled, y_resampled = BorderlineSMOTE(kind='borderline-1').fit_resample(X_train, y_train)
elif choice == 5:
    '''over-sampling,BorderlineSMOTE'''
    X_resampled, y_resampled = BorderlineSMOTE(kind='borderline-2').fit_resample(X_train, y_train)
elif choice == 6:
    '''down-sampling,原型生成,ClusterCentroids'''
    cc = ClusterCentroids(random_state=0)
    X_resampled, y_resampled = cc.fit_resample(X_train, y_train)
elif choice == 7:
    '''down-sampling,原型选择,直接随机采样'''
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
elif choice == 8:
    '''down-sampling,原型选择,清洗数据,Tomek's links'''
    X_resampled, y_resampled = TomekLinks().fit_resample(X_train, y_train)
elif choice == 9:
    '''down-sampling,原型选择,清洗数据,EdittedNearestNeighbours'''
    X_resampled, y_resampled = EditedNearestNeighbours(kind_sel='mode').fit_resample(X_train, y_train)
elif choice == 10:
    '''过采样与下采样结合,SMOTE+Tomek's links'''
    smote_tomek = SMOTETomek(random_state=0)
    X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
elif choice == 11:
    '''过采样与下采样结合,SMOTE+EdittedNearestNeighbours'''
    smote_enn = SMOTEENN(random_state=0)
    X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)

train_model(X_resampled, y_resampled, X_test, y_test,'LogisticRegression')
train_model(X_resampled, y_resampled, X_test, y_test,'Dtree')
train_model(X_resampled, y_resampled, X_test, y_test,'Rforest')
train_model(X_resampled, y_resampled, X_test, y_test,'Gboost')
train_model(X_resampled, y_resampled, X_test, y_test,'SVM')
train_model(X_resampled, y_resampled, X_test, y_test,'MLP')