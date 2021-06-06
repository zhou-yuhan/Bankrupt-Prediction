# Author: 周裕涵

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from collections import namedtuple


class ClassifyModel:
    # selected features, 0(bankrupted?) is label
    features = [0, 1, 2, 3, 68, 5, 6, 7, 9, 12, 13, 78, 16, 18, 85, 89, 90, 91, 92, 94, 31, 32, 33, 34, 35, 36, 37, 41, 42, 54, 61]
    Metrics = namedtuple('Metrics', ['accuracy', 'precision', 'recall', 'f1'])

    def __init__(self, filename="D:\\Intro to Machine Learning\\Bankrupt-Prediction\\data.csv"):
        self.methods = {}

        self.rawdata = pd.read_csv(filename)
        print('Original Data:')
        print(self.rawdata.info())

        y = self.rawdata["Bankrupt?"]
        X = self.rawdata.drop(columns=["Bankrupt?"], axis = 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=.2, shuffle=True)

        #缩放，standarlization
        scaler = StandardScaler()
        scaler.fit(X)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_test_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average="binary")
        print("Method: Origin Random Forest")
        print("accuracy={0:.4f}, precision={1:.4f}, recall={2:.4f}, f1_score={3:.4f}".format\
              (accuracy, precision, recall, f1))
        self.methods['Origin'] = ClassifyModel.Metrics(accuracy, precision, recall, f1)

        ########################################################################################
        # 读取数据
        self.data = pd.read_csv(filename, usecols=ClassifyModel.features)
        print('Unsampled Data:')
        print(self.data.info())
        # 准备数据 X:features y:labels
        # test_size表示测试集大小与总数据集大小之比，可以调整，现在是0.2
        self.y = self.data["Bankrupt?"]
        self.X = self.data.drop(columns=["Bankrupt?"], axis = 1)
        X_train, self.X_test, y_train, self.y_test = train_test_split(self.X, self.y, random_state=0, test_size=.2, shuffle=True)
        #缩放，standarlization
        scaler = StandardScaler()
        scaler.fit(self.X)
        X_train = scaler.transform(X_train)
        self.X_test = scaler.transform(self.X_test)
        # imbalanced learning
        self.X_resampled, self.y_resampled = EditedNearestNeighbours(kind_sel='mode').fit_resample(X_train, y_train)
    
    def _corr_skew(self, X):
        s = X.skew().reset_index().rename(columns = {0:'skew'})

        pos = list(s[s['skew']>=1]['index'].values)
        neg = list(s[s['skew']<=-1]['index'].values)

        X[pos] = (X[pos]+1).apply(np.log)
        X[neg] = (X[neg])**3
        return X

    def DecisionTree(self):
        model = DecisionTreeClassifier(random_state=0).fit(self.X_resampled, self.y_resampled)
        y_test_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_test_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(self.y_test, y_test_pred, average="binary")
        print("Method: Decision Tree")
        print("accuracy={0:.4f}, precision={1:.4f}, recall={2:.4f}, f1_score={3:.4f}".format\
              (accuracy, precision, recall, f1))
        self.methods['DecisionTree'] = ClassifyModel.Metrics(accuracy, precision, recall, f1)

    def RandomForest(self):
        model = RandomForestClassifier(n_estimators=10, random_state=0).fit(self.X_resampled, self.y_resampled)
        y_test_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_test_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(self.y_test, y_test_pred, average="binary")
        print("Method: Random Forest")
        print("accuracy={0:.4f}, precision={1:.4f}, recall={2:.4f}, f1_score={3:.4f}".format\
              (accuracy, precision, recall, f1))
        self.methods['RandomForest'] = ClassifyModel.Metrics(accuracy, precision, recall, f1)

    def SVM(self):
        model = SVC().fit(self.X_resampled, self.y_resampled)
        y_test_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_test_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(self.y_test, y_test_pred, average="binary")
        print("Method: SVM")
        print("accuracy={0:.4f}, precision={1:.4f}, recall={2:.4f}, f1_score={3:.4f}".format\
              (accuracy, precision, recall, f1))
        self.methods['SVM'] = ClassifyModel.Metrics(accuracy, precision, recall, f1)
    
    def BernoulliNB(self):
        X = self._corr_skew(self.X)
        y = self.y
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=.2, shuffle=True)
        #缩放，standarlization
        scaler = StandardScaler()
        scaler.fit(X)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        # imbalanced learning
        X_resampled, y_resampled = EditedNearestNeighbours(kind_sel='mode').fit_resample(X_train, y_train) 

        model = BernoulliNB(alpha = 10).fit(X_resampled, y_resampled) # alpha = 10 gets maximum AUC
        y_test_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_test_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average="binary")
        print("Method: Bernoulli Naive Bayes")
        print("accuracy={0:.4f}, precision={1:.4f}, recall={2:.4f}, f1_score={3:.4f}".format\
              (accuracy, precision, recall, f1))
        self.methods['BernoulliNB'] = ClassifyModel.Metrics(accuracy, precision, recall, f1)
    
    def GaussianNB(self):
        X = self._corr_skew(self.X)
        y = self.y
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=.2, shuffle=True)
        #缩放，standarlization
        scaler = StandardScaler()
        scaler.fit(X)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        # imbalanced learning
        X_resampled, y_resampled = EditedNearestNeighbours(kind_sel='mode').fit_resample(X_train, y_train) 

        model = GaussianNB().fit(X_resampled, y_resampled) # alpha = 10 gets maximum AUC
        y_test_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_test_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average="binary")
        print("Method: Gaussian Naive Bayes")
        print("accuracy={0:.4f}, precision={1:.4f}, recall={2:.4f}, f1_score={3:.4f}".format\
              (accuracy, precision, recall, f1))
        self.methods['GaussianNB'] = ClassifyModel.Metrics(accuracy, precision, recall, f1)
    
    def plot(self):
        labels = list(self.methods.keys())
        accus= [self.methods[method].accuracy for method in labels]
        precisions = [self.methods[method].precision for method in labels]
        recalls = [self.methods[method].recall for method in labels]
        f1s = [self.methods[method].f1 for method in labels]

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
        width_1 = 0.4
        ax1.bar(np.arange(len(accus)), accus ,width=width_1,tick_label=labels, fc='r')
        for x, y in enumerate(accus):
            ax1.text(x - 0.2, y, "{0:.2f}".format(y))
        ax1.set_title('Comparing model accuracy')

        ax2.bar(np.arange(len(precisions)), precisions ,width=width_1,tick_label=labels, fc='g')
        for x, y in enumerate(precisions):
            ax2.text(x - 0.2, y, "{0:.2f}".format(y))
        ax2.set_title('Comparing model precision rate')

        ax3.bar(np.arange(len(recalls)), recalls, width=width_1, tick_label=labels, fc='b')
        for x, y in enumerate(recalls):
            ax3.text(x - 0.2, y, "{0:.2f}".format(y))
        ax3.set_title('Comparing model recall rate')

        ax4.bar(np.arange(len(f1s)), f1s, width=width_1, tick_label=labels, fc='y')
        for x, y in enumerate(f1s):
            ax4.text(x - 0.2, y, "{0:.4f}".format(y))
        ax4.set_title('Comparing model f1 score')
        plt.tight_layout()
        plt.show() 