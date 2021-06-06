# Author: 周裕涵

import numpy as np
import matplotlib.pyplot as plt
from model import ClassifyModel


def inbalancedPlot(method, new_accu, new_f1):
    labels = ["L-Regression", "D-Tree", "R-Forest", "G-Boost", "SVM", "MLP"]
    origin_accu = [97.07, 95.45, 97.07, 97.14, 97.14, 96.99]
    origin_f1 = [0.2593, 0.3111, 0.1667, 0.3607, 0.0488, 0.3279]

    fig, (ax1, ax2) = plt.subplots(2, 1)
    width_1 = 0.4
    ax1.bar(np.arange(len(origin_accu)), origin_accu ,width=width_1,tick_label=labels,label="origin")
    for x, y in enumerate(origin_accu):
        ax1.text(x - 0.2, y, "{0:.2f}".format(y))
    ax1.bar(np.arange(len(new_accu))+width_1, new_accu, width=width_1,tick_label=labels,label=method)
    for x, y in enumerate(new_accu):
        ax1.text(x + 0.2, y, "{0:.2f}".format(y))
    ax1.legend()
    ax1.set_title('Comparing Model accuracy before and after using method {}'.format(method))

    ax2.bar(np.arange(len(origin_f1)), origin_f1 ,width=width_1,tick_label=labels,label="origin")
    for x, y in enumerate(origin_f1):
        ax2.text(x - 0.3, y, "{0:.4f}".format(y))
    ax2.bar(np.arange(len(new_f1))+width_1, new_f1, width=width_1,tick_label=labels,label=method)
    for x, y in enumerate(new_f1):
        ax2.text(x + 0.1, y, "{0:.4f}".format(y))
    ax2.legend()
    ax2.set_title('Comparing Model F1 score before and after using method {}'.format(method))
    plt.tight_layout()
    plt.show() 

def inbalancedResult():
    methods = ['Random', 'SMOTE', 'ADASYN', 'Cluster', 'Tomeks Links', 'Edited NN', 'SMOTE+Tomek']
    new_accus = [
        [86.51, 96.11, 97.07, 92.08, 87.61, 96.06],
        [87.98, 93.99, 95.75, 93.33, 88.64, 96.41],
        [86.07, 93.55, 95.53, 92.30, 89.22, 96.41],
        [79.55, 59.16, 79.25, 64.59, 75.73, 81.23],
        [97.14, 95.60, 96.99, 97.14, 97.14, 96.77],
        [97.21, 95.09, 97.21, 97.14, 97.14, 96.70],
        [87.98, 93.55, 96.33, 93.48, 88.71, 96.11]
    ]

    new_f1s = [
        [0.2459, 0.3291, 0.3333, 0.3333, 0.2489, 0.2703],
        [0.2743, 0.2931, 0.3830, 0.2546, 0.2654, 0.3288],
        [0.2460, 0.2903, 0.3297, 0.3046, 0.2613, 0.3467],
        [0.2051, 0.1059, 0.1891, 0.1329, 0.1704, 0.1950],
        [0.2909, 0.3517, 0.1961, 0.3810, 0.0488, 0.3333],
        [0.3667, 0.3093, 0.2692, 0.4179, 0.0488, 0.3478],
        [0.2743, 0.2281, 0.4186, 0.3776, 0.2667, 0.2993] 
    ]

    for i in range(len(methods)):
        inbalancedPlot(methods[i], new_accus[i], new_f1s[i])

def finalResult():
    models = ClassifyModel()
    models.DecisionTree()
    models.RandomForest()
    models.SVM()
    models.BernoulliNB()
    models.GaussianNB()
    models.plot()

def NNCompareResult():
    labels = ['accuracy', 'precision', 'recall', 'f1']
    bayes = [0.96, 0.43, 0.50, 0.46]
    nn = [0.97, 0.48, 0.50, 0.49]
    fig, ax1= plt.subplots(1, 1)
    width_1 = 0.4
    ax1.bar(np.arange(len(bayes)), bayes ,width=width_1,tick_label=labels,label="GaussianNB")
    for x, y in enumerate(bayes):
        ax1.text(x - 0.2, y, "{0:.2f}".format(y))
    ax1.bar(np.arange(len(nn))+width_1, nn, width=width_1,tick_label=labels,label="NeuralNetwork")
    for x, y in enumerate(nn):
        ax1.text(x + 0.2, y, "{0:.2f}".format(y))
    ax1.legend()
    ax1.set_title('Comparing model between Bayes and Neural Network')
    plt.show()


if __name__ == '__main__':
    # inbalancedResult()
    # finalResult()
    # NNCompareResult()
    pass