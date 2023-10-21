'''
:@Author: Dizzying
:@Date: 10/19/2023, 3:06:20 PM
:@LastEditors: Dizzying
:@LastEditTime: 10/19/2023, 3:06:20 PM
:Description: 
:Copyright: Copyright (©)}) 2023 XuWei. All rights reserved.
'''
import torch
from torch import nn
from d2l import torch as d2l

# 按照每次256个数据批量来获取数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
# 得到训练模型
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

# 初始化
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)

# 定义损失函数
loss = nn.CrossEntropyLoss(reduction='none')

# 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

# 跑十次
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)