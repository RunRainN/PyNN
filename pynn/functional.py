"""
文件：functional.py
作者：RunRain
日期：2023/1/14
描述：运算函数实现
"""

import numpy as np


def sigmoid(x):
    """sigmoid函数"""
    return 1 / (1 + np.exp(-x))


def relu(x):
    """ReLU函数"""
    return np.maximum(0, x)


def softmax(x):
    """Softmax函数"""
    x_shift = x - np.max(x, axis=-1, keepdims=True)
    x_shift_exp = np.exp(x_shift)
    return x_shift_exp / np.sum(x_shift_exp, axis=-1, keepdims=True)
