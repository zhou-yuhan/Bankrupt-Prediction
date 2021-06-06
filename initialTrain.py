# Author: 张柏舟

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Read the voice dataset
mydata = pd.read_csv("D:\\Intro to Machine Learning\\Bankrupt-Prediction\\data.csv")
print(mydata.info()) #一共6819个样本

bankrupted = mydata.loc[mydata['Bankrupt?'] == 1]
print(bankrupted) #破产的银行只有其中的200+个样本


mydata_train, mydata_test = train_test_split(mydata, random_state=0, test_size=.2) #训练集：测试集，8：2
scaler = StandardScaler()
scaler.fit(mydata_train.iloc[:, 0:-1])
X_train = scaler.transform(mydata_train.iloc[:, 0:-1])
X_test = scaler.transform(mydata_test.iloc[:, 0:-1])
y_train = list(mydata_train['Bankrupt?'].values)
y_test = list(mydata_test['Bankrupt?'].values)

# Train decision tree model
tree = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
print("Decision Tree")
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.4f}".format(tree.score(X_test, y_test)))
# Train random forest model
forest = RandomForestClassifier(n_estimators=5, random_state=0).fit(X_train, y_train)
print("Random Forests")
print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set: {:.4f}".format(forest.score(X_test, y_test)))
# Train gradient boosting model
gbrt = GradientBoostingClassifier(random_state=0).fit(X_train, y_train)
print("Gradient Boosting")
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.4f}".format(gbrt.score(X_test, y_test)))
# Train support vector machine model
svm = SVC().fit(X_train, y_train)
print("Support Vector Machine")
print("Accuracy on training set: {:.3f}".format(svm.score(X_train, y_train)))
print("Accuracy on test set: {:.4f}".format(svm.score(X_test, y_test)))

# Train neural network model
mlp = MLPClassifier(random_state=0).fit(X_train, y_train)
print("Multilayer Perceptron")
print("Accuracy on training set: {:.3f}".format(mlp.score(X_train, y_train)))
print("Accuracy on test set: {:.4f}".format(mlp.score(X_test, y_test)))



