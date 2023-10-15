# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 21:03:58 2023

@author: Dizzying
"""
import numpy as np
import torch
from torch import nn
from torch.utils import data
from d2l import torch as d2l

'''
###########函数库的解释###########

from torch.utils import data
PyTorch 提供了两个数据基类： torch.utils.data.DataLoader 和 torch.utils.data.Dataset。
允许你使用预加载的数据集以及你自己的数据集。 
Dataset 存储样本和它们相应的标签，
DataLoader 在 Dataset 基础上添加了一个迭代器，迭代器可以迭代数据集，以便能够轻松地访问 Dataset 中的样本

from torch import nn
nn为Neural Network的意思，其中包含了很多的神经网络当中的方法和接口


'''

#---------------函数库---------------#
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """
    构造一个PyTorch数据迭代器
    """
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
#---------------函数库---------------#

#---------------线性回归具体实现---------------#

# 生成数据集，原理与线性回归从0开始相同
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# 批量处理大小
batch_size = 10

# 创建数据迭代器
data_iter = load_array((features, labels), batch_size)

# 首先定义一个模型变量net，它是一个Sequential类的实例。 Sequential类将多个层串联在一起。 当给定输入数据时，Sequential实例将数据传入到第一层， 然后将第一层的输出作为第二层的输入
# 创建一个线性层，输入数据的维度为2，输出数据的维度为1，存放在Sequential里面
net = nn.Sequential(nn.Linear(2, 1))

# 初始化参数
# 通过net[0]选择网络中的第一个图层， 然后使用weight.data和bias.data方法访问参数，使用替换方法normal_和fill_来重写参数值
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 计算均方误差使用的是MSELoss类，也称为平方L2范数。默认情况下，它返回所有样本损失的平均值
loss = nn.MSELoss()

# 小批量随机梯度下降算法是一种优化神经网络的标准工具
# 当实例化一个SGD实例时，要指定优化的参数 （可通过net.parameters()从我们的模型中获得）以及优化算法所需的超参数字典。 
# 小批量随机梯度下降的lr值设置为0.03
trainer = torch.optim.SGD(net.parameters(), lr=0.01)

# 循环次数为3
num_epochs = 10

#开始循环
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)