"""
文件：optim.py
作者：RunRain
日期：2023/1/14
描述：优化器定义
"""

import numpy as np


class Optim(object):
    """优化器基类"""

    def __init__(self, net_params, lr=1e-2):
        self.net_params = net_params
        self.lr = lr

    def zero_grad(self):
        for params in self.net_params:
            params.dW *= 0
            params.db *= 0

    def step(self):
        pass


class SGD(Optim):
    """带momentum的SGD优化器"""

    def __init__(self, net_params, lr=1e-2, momentum=0):
        super(SGD, self).__init__(net_params, lr)
        self.momentum = momentum
        self.v_W_list = [np.zeros(params.W.shape) for params in self.net_params]
        self.v_b_list = [np.zeros(params.b.shape) for params in self.net_params]

    def step(self):
        for i, params in enumerate(self.net_params):
            self.v_W_list[i] = self.momentum * self.v_W_list[i] - self.lr * params.dW
            self.v_b_list[i] = self.momentum * self.v_b_list[i] - self.lr * params.db
            params.W += self.v_W_list[i]
            params.b += self.v_b_list[i]


class Adagrad(Optim):
    """Adagrad优化器"""

    def __init__(self, net_params, lr=1e-2, eps=1e-10):
        super(Adagrad, self).__init__(net_params, lr)
        self.eps = eps
        self.h_W_list = [np.zeros(params.W.shape) for params in self.net_params]
        self.h_b_list = [np.zeros(params.b.shape) for params in self.net_params]

    def step(self):
        for h_W, h_b, params in zip(self.h_W_list, self.h_b_list, self.net_params):
            h_W += np.square(params.dW)
            h_b += np.square(params.db)
            params.W -= self.lr * params.dW / np.sqrt(h_W + self.eps)
            params.b -= self.lr * params.db / np.sqrt(h_b + self.eps)


class Adam(Optim):
    """Adam优化器"""

    def __init__(self, net_params, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8):
        super(Adam, self).__init__(net_params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m_W_list = [np.zeros(params.W.shape) for params in self.net_params]
        self.m_b_list = [np.zeros(params.b.shape) for params in self.net_params]
        self.v_W_list = [np.zeros(params.W.shape) for params in self.net_params]
        self.v_b_list = [np.zeros(params.b.shape) for params in self.net_params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, params in enumerate(self.net_params):
            self.m_W_list[i] = self.beta1 * self.m_W_list[i] + (1 - self.beta1) * params.dW
            self.m_b_list[i] = self.beta1 * self.m_b_list[i] + (1 - self.beta1) * params.db
            self.v_W_list[i] = self.beta2 * self.v_W_list[i] + (1 - self.beta2) * np.square(params.dW)
            self.v_b_list[i] = self.beta2 * self.v_b_list[i] + (1 - self.beta2) * np.square(params.db)
            step_size = self.lr * np.sqrt(1 - np.power(self.beta2, self.t)) / (1 - np.power(self.beta1, self.t))
            params.W -= step_size * self.m_W_list[i] / (np.sqrt(self.v_W_list[i]) + self.eps)
            params.b -= step_size * self.m_b_list[i] / (np.sqrt(self.v_b_list[i]) + self.eps)
