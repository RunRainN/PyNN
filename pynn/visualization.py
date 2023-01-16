"""
文件：visualization.py
作者：RunRain
日期：2023/1/13
描述：训练过程可视化实现
"""

import matplotlib.pyplot as plt
import time


class Graph(object):
    """图窗类"""

    def __init__(self, row, col, xrange=None, yrange=None, titles=None):
        self.row = row
        self.col = col
        self.count = 0
        self.data_x = []
        self.data_y = []
        plt.ion()
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.subplots(row, col)
        for i in range(row):
            for j in range(col):
                idx = i * col + j
                plt.subplot(row, col, idx + 1)
                plt.xlim(xrange) if xrange else None
                # plt.xticks([], [])
                plt.ylim(yrange) if yrange else None
                # plt.yticks([], [])
                plt.title(titles[idx]) if titles else None
                self.data_y.append([])  # data_y为二维列表
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def __call__(self, *args: float):
        self.count += 1
        self.data_x.append(self.count)
        for i in range(self.row):
            for j in range(self.col):
                idx = i * self.col + j
                self.data_y[idx].append(args[idx])
                self.ax[i, j].plot(self.data_x, self.data_y[idx], 'b.-')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def save(self, path):
        self.fig.savefig(path)

# g = Graph(2, 2, xrange=(0, 50), titles=('train_loss', 'train_accuracy', 'valid_loss', 'valid_accuracy'))
# for i in range(50):
#     time.sleep(5)  # 小bug：等待时图窗显示未响应，但不影响程序运行，图窗可以正常绘图
#     g(0.5, 0.5, 0.5, 0.5)
