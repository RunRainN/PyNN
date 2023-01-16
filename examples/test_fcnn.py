"""
文件：test_fcnn.py
作者：RunRain
日期：2023/1/14
描述：FCNN测试
"""

import sys
import numpy as np

sys.path.append('..')
sys.path.append('../pynn')

from pynn import nn
from pynn.net import Net
from pynn.optim import SGD


class FCNN(Net):
    def __init__(self):
        super(FCNN, self).__init__()
        self.struct = nn.Sequential(
            nn.Linear(2, 3),
            # nn.Sigmoid()
            nn.ReLU()
        )
        self.criterion = nn.MSELoss()


def test_fcnn():
    # 实际的权重和偏置
    W = np.array([[1, 2, 3],
                  [4, 5, 6]])
    b = np.array([7, 8, 9])
    # 产生训练样本
    x_data = np.random.randn(500, 2)
    y_data = np.dot(x_data, W) + b
    # y_data = 1 / (1 + np.exp(-y_data))  # sigmoid
    y_data = np.maximum(0, y_data)  # relu

    def next_sample(size=1):
        idx = np.random.randint(500)
        return x_data[idx:idx + size], y_data[idx:idx + size]

    model = FCNN()
    sgd = SGD(model.parameters(), lr=1e-1)
    batch_size = 4
    i = 0
    loss = 1
    while loss > 1e-15:
        x, y_true = next_sample(batch_size)  # 获取当前样本
        # 前向传播
        y = model(x)
        # 梯度清零
        sgd.zero_grad()
        # 计算损失
        loss = model.criterion(y, y_true)
        # 反向传播
        model.backward()
        # 更新梯度
        sgd.step()
        # 更新迭代次数
        i += 1
        if i % 10000 == 0:
            print("\n迭代{}次，当前loss:{}, 当前权重:{},当前偏置{}"
                  .format(i, loss, model.struct.params[0].W, model.struct.params[0].b))
    print("\n迭代{}次，当前loss:{}, 当前权重:{},当前偏置{}"
          .format(i, loss, model.struct.layers[0].W, model.struct.params[0].b))


test_fcnn()
