"""
文件：visualization.py
作者：RunRain
日期：2023/1/12
描述：网络层定义
"""

import numpy as np

import functional as F


class Layer(object):
    """层基类"""

    def __init__(self):
        self.input = None
        self.output = None
        self.grad = None  # dinput(∂L/∂input)

    def forward(self, *args):
        pass

    def backward(self, *args):
        pass

    def __call__(self, *args):
        return self.forward(*args)


class Parameters(object):
    """权重参数类"""

    def __init__(self, W, dW, b, db):
        self.W = W
        self.dW = dW
        self.b = b
        self.db = db


class Linear(Layer):
    """全连接层/线性层/Affine层"""

    def __init__(self, dim_in, dim_out):
        super(Linear, self).__init__()
        self.W = np.random.randn(dim_in, dim_out) * np.sqrt(2 / dim_in)
        self.b = np.zeros(dim_out)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.params = Parameters(self.W, self.dW, self.b, self.db)

    def forward(self, x):
        self.input = x
        self.output = np.dot(self.input, self.W) + self.b  # z_l = a_l-1·W + b
        return self.output

    def backward(self, grad):
        self.dW += np.dot(self.input.T, grad)  # grad = dout(dz_l)
        # 这里不能使用‘=’直接赋值，原因如下：
        # self.db和self.params.db指代的是同一nd对象，使用‘=’给其中某一变量赋值会使这个变量的指代对象发生改变，导致这两个变量不再指代同一对象，
        # 而‘+=’只对指代的nd对象的值进行修改，但这就需要保证反向传播前梯度为0，因此在反向传播前需要执行梯度清零操作
        self.db += np.sum(grad, axis=0)
        self.grad = np.dot(grad, self.W.T)  # grad = dinput(da_l-1)
        return self.grad


class Sigmoid(Layer):
    """Sigmoid激活函数层"""

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        self.input = x
        self.output = F.sigmoid(x)
        return self.output

    def backward(self, grad):
        self.grad = grad * self.output * (1 - self.output)
        return self.grad


class ReLU(Layer):
    """ReLU激活函数层"""

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.input = x
        self.output = F.relu(x)
        return self.output

    def backward(self, grad):
        self.grad = np.copy(grad)
        self.grad[self.input > 0] *= 1
        self.grad[self.input <= 0] *= 0
        return self.grad


# Sequential可以看成是所有层组成的由输入到输出的一层
class Sequential(Layer):
    """层容器"""

    def __init__(self, *layers):
        super(Sequential, self).__init__()
        self.layers = []
        self.params = []
        for layer in layers:
            self.layers.append(layer)
            if isinstance(layer, Linear):
                self.params.append(layer.params)
        self.input = self.layers[0].input
        self.output = self.layers[-1].output
        self.grad = self.layers[0].grad

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, grad):
        for idx in range(len(self.layers) - 1, -1, -1):
            grad = self.layers[idx].backward(grad)

    def add_layer(self, layer):
        self.layers.append(layer)
        if isinstance(layer, Linear):
            self.params.append(layer.params)


class MSELoss(Layer):
    """均方误差损失函数层"""

    def __init__(self):
        super(MSELoss, self).__init__()
        self.true = None

    def forward(self, pred, true):
        self.input = pred
        self.true = true
        self.output = np.mean(np.sum(0.5 * np.square(pred - true), axis=-1))
        return self.output

    def backward(self):
        self.grad = (self.input - self.true) / self.input.shape[0]
        return self.grad


class CrossEntropyLoss(Layer):
    """交叉熵损失函数层（带Softmax）"""

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.true = None
        self.prob = None

    def forward(self, pred, true):
        self.input = pred
        self.true = true
        self.prob = F.softmax(pred)
        self.output = -np.mean(np.sum(true * np.log(self.prob), axis=-1))
        return self.output

    def backward(self):
        self.grad = (self.prob - self.true) / self.input.shape[0]
        return self.grad
