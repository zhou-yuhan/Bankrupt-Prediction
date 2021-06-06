# Author: 鲁琦琨

# 构造了一个三层的全连接神经网络，由于二分类问题本身不复杂，epoch=50时准确率就能收敛到99%
import pandas as pd
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN
from sklearn.metrics import classification_report

company = pd.read_csv('D:\\Intro to Machine Learning\\Bankrupt-Prediction\\data.csv')


p = 1-(len(company[company['Bankrupt?']==1])/len(company))
X = company.drop('Bankrupt?', axis = 1).reset_index(drop = True)
y = company['Bankrupt?'].reset_index(drop = True)
X_resample, y_resample = X, y # ADASYN().fit_resample(X, y) 使用ADASYN解决数据不平衡的问题，作为对照组暂不采用

# 预处理数据
c = X_resample.columns
scaler = StandardScaler()
X_resample = pd.DataFrame(scaler.fit_transform(X_resample), columns = X_resample.columns)
X_resample = np.float32(X_resample)
y_resample = np.float32(y_resample)
X_train, X_test,y_train,y_test = train_test_split(X_resample,y_resample,test_size = 0.2, random_state=49)
accs = []

class CreditDataset(Dataset):
    def __init__(self, X, y):
        self.labels = y
        self.X = X

    def __getitem__(self, index):
        return self.labels[index], self.X[index]

    def __len__(self):
        return len(self.X)

# 准备数据集
train_dataloader = DataLoader(CreditDataset(X_train,y_train), batch_size = 64)
test_dataloader = DataLoader(CreditDataset(X_test,y_test), batch_size = 64)


# 定义模型类
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(95, 95),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(95, 95),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(95, 2)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

# 建立训练函数
def train_loop(dataloader, model, loss_fn, optimizer):
    for y, x in dataloader:
        y_pred = model(x)
        loss = loss_fn(y_pred, y.type(torch.long))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 建立测试函数
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for y, x in dataloader:
            y_pred = model(x)
            test_loss += loss_fn(y_pred, y.type(torch.long))
            correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    accs.append(correct)
    print(f"Accuracy: {(correct * 100):>8f}%, Avg loss: {test_loss:>8f} \n")

# 训练过程
epochs = 200
model = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
fig, ax = plt.subplots()
for i in range(epochs):
    print(f"Epoch {i + 1}")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
ax.plot(accs)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()


# 结果展示
X_final = company[c]
y_pred_final = model(torch.from_numpy(np.float32(X_final.values)))
print(classification_report(y,y_pred_final.argmax(1)))

