# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 19:26:48 2023

@author: Dizzying
"""

import math
import time
import numpy as np
import torch
import random

#---------------函数库---------------#
def synthetic_data(w, b, num_examples):  #@save
    """
    生成y=Xw+b噪声
    w&b:参数 
    num_examples: 需要的样本个数
    """
    # torch.normal：返回一个张量，包含从给定参数means,std的离散正态分布中抽取随机数
    # X为均值为0，方差为1的随机数，样本个数为num_examples，列数为w的长度
    X = torch.normal(0, 1, (num_examples, len(w)))

    # torch.matmul是tensor的乘法
    y = torch.matmul(X, w) + b
    
    # 加入均值为0，方差为0.01，形状为y的随机噪声
    y += torch.normal(0, 0.01, y.shape)

    # 输出y的形状
    print(y.shape)

    # y.reshape((-1, 1))返回一个不知道多少行，但是列数为1的张量
    return X, y.reshape((-1, 1))

def data_iter(batch_size, features, labels):
    """
    batch_size: 单次循环数据量
    features: 矩阵X
    labels: 向量Y
    """
    # num_examples为样本的个数
    num_examples = len(features)
    
    # indices为样本中的索引
    indices = list(range(num_examples))
    
    # random.shuffle为将list随机打乱，改变原list
    random.shuffle(indices)
    
    #对所有的样本数量，按照步长为batch_size的大小进行遍历
    for i in range(0, num_examples, batch_size):
        
        # 从indices中拿出对应的个数的索引，并转化为张量
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        
        # yield类似于return 但是不会中止程序
        yield features[batch_indices], labels[batch_indices]

def linreg(X, w, b):  #@save
    """
    定义线性回归模型
    """
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):  #@save
    """
    均方损失
    """
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):  #@save
    """
    小批量随机梯度下降
    """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
#---------------函数库---------------#

#---------------线性回归具体实现---------------#

# true_w和true_b为真实的参数
true_w = torch.tensor([2, -3.4])
true_b = 4.2

# features为1000行2列的矩阵， labels为1000行的列向量，
features, labels = synthetic_data(true_w, true_b, 1000)

# 将features转化为numpy的格式并输出展示
np_features = features.detach().numpy()
print(np_features)

# 定义每次循环的数据量
batch_size = 10

# 初始化需要学习的w和b参数
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 学习率
lr = 0.01

# 循环次数
num_epochs = 10

# 定义模型
net = linreg

#定义损失函数
loss = squared_loss

#开始循环
for epoch in range(num_epochs):
    
    #从数据集中循环随机抽取batch_size个数据
    for X, y in data_iter(batch_size, features, labels):
        #损失函数
        l = loss(net(X, w, b), y)  
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        # 使用参数的梯度更新参数
        sgd([w, b], lr, batch_size)  
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')











