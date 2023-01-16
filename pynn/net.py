"""
文件：net.py
作者：RunRain
日期：2023/1/12
描述：网络模型定义
"""

import numpy as np


class Net(object):
    """网络模型基类"""

    def __init__(self):
        self.struct = None
        self.criterion = None

    def forward(self, x):
        return self.struct.forward(x)

    def backward(self):
        grad = self.criterion.backward()
        self.struct.backward(grad)

    def parameters(self):
        return self.struct.params

    def __call__(self, x):
        return self.forward(x)

    def save(self, path):
        params_dict = {}
        for i, params in enumerate(self.struct.params):
            params_dict.update({'W_' + str(i + 1): params.W})
            params_dict.update({'b_' + str(i + 1): params.b})
        np.savez(path, **params_dict)

    def load(self, path):
        weights = np.load(path)
        for i, params in enumerate(self.struct.params):
            params.W *= 0
            params.W += weights['W_' + str(i + 1)]
            params.b *= 0
            params.b += weights['b_' + str(i + 1)]
