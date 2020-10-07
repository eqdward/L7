# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 11:01:33 2020

@author: yy
"""


"""0. 数据加载"""
from sklearn.datasets import load_boston

boston = load_boston()
X = boston['data']
y = boston['target'].reshape(-1,1)


"""1. 数据预处理"""
# 特征工程——数据标准化
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X = ss.fit_transform(X)

# 数据格式转化
import torch
X = torch.from_numpy(X).type(torch.FloatTensor) 
y = torch.from_numpy(y).type(torch.FloatTensor) 

# 数据集分割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 666)


"""2. 构造神经网络"""
from torch import nn

input_dim = X.shape[1]   # 输入层维度
hidden_dim1 = 10   # 隐藏1层维度
hidden_dim2 = 12   # 隐藏2层维度
hidden_dim3 = 8   # 隐藏3层维度
output_dim = 1   # 输出层维度

# 定义模型
model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim1),
        nn.ReLU(),
        nn.Linear(hidden_dim1, hidden_dim2),
        nn.ReLU(),
        nn.Linear(hidden_dim2, hidden_dim3),
        nn.ReLU(),
        nn.Linear(hidden_dim3, output_dim)
    )

# 定义优化器和损失函数
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


"""3. 模型训练"""
max_epoch = 300
iter_loss = []
for i in range(max_epoch):
    # 前向传播计算预测值
    y_pred = model(X_train)
    # 计算损失loss
    loss = criterion(y_pred, y_train)
    # 保存当前迭代的loss
    iter_loss.append(loss.item())
    # 刷新梯度
    optimizer.zero_grad()
    # 反向传递
    loss.backward()
    # 权重调整
    optimizer.step()
    

"""4. 模型测试"""
output = model(X_test)
predict_list = output.detach().numpy()


"""5. 绘制loss变化趋势"""
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(max_epoch)
y = np.array(iter_loss)

plt.plot(x, y)
plt.title('Loss value changing in iterations')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()


"""6. 比较真实值与预测值"""
x = np.arange(X_test.shape[0])
y1 = np.array(predict_list)
y2 = np.array(y_test)

plt.figure(figsize=(20,10))
line1 = plt.scatter(x, y1, c='red')
line2 = plt.scatter(x, y2, c='blue')
for i in range(len(x)):
    plt.plot([x[i], x[i]], [y1[i], y2[i]], c='orange')
plt.legend([line1, line2], ['Predict Val', 'True Val'], loc=0)
plt.title('True VS. Prediction')
plt.ylabel('House Price')
plt.show()
