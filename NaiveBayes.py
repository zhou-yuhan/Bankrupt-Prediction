import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import *
from sklearn.feature_selection import *
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.ensemble import *
from sklearn.decomposition import PCA
from sklearn.naive_bayes import BernoulliNB

# this code help in displaying complete block of data rather than ... in the columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

FOLDS = 5

data = pd.read_csv('D:\\Intro to Machine Learning\\Bankrupt-Prediction\\data.csv')
data.head()

data.describe()


#show the histogram of the dataset
data.hist(figsize=(80,40))
plt.show()



#finds the skewness of each column.
def corr_skew(X):
    s = X.skew().reset_index().rename(columns = {0:'skew'})

    pos = list(s[s['skew']>=1]['index'].values)
    neg = list(s[s['skew']<=-1]['index'].values)

    X[pos] = (X[pos]+1).apply(np.log)
    X[neg] = (X[neg])**3
    return X



def pred_stratified(X,y):
    X = X.values
    y = y.values

    skf = StratifiedKFold(n_splits=FOLDS)
    aucs = []
    fig, ax = plt.subplots()

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = BernoulliNB(alpha = 10)
        model.fit(X_train,y_train)

        yxgb = model.predict(X_test)
        plot_roc_curve(model, X_test, y_test, ax=ax)
        aucs.append(roc_auc_score(y_true=y_test,y_score=yxgb))
    plt.show()      
    return sum(aucs)/5

if __name__ == '__main__':
    y  = data['Bankrupt?']
    X  = data.drop(['Bankrupt?'],axis=1)
    X  = corr_skew(X)

    print('Correlation: \n+1 means direct, i.e increase in column will lead increase in label value.\n-1 for just the opposite.）' )
    print(data.corrwith(data['Bankrupt?']))
    print('Naive Bayes:')
    print(pred_stratified(X,y))


