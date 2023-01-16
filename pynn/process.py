"""
文件：process.py
作者：RunRain
日期：2023/1/13
描述：训练、验证、测试过程相关实现
"""

import numpy as np


# train_loader训练集批数据迭代器，model网络模型，optimizer优化方法，classification是否为分类模型
def train_model(train_loader, model, optimizer, classification: bool):
    """模型训练函数"""
    train_loss = []
    train_accuracy = [] if classification else None
    for i, data in enumerate(train_loader):
        inputs, labels = data[0], data[1]
        outputs = model(inputs)  # 前向传播
        loss = model.criterion(outputs, labels)  # 计算该批数据的损失
        train_loss.append(loss)
        if classification:
            preds = np.argmax(outputs, axis=-1)  # 获取该批数据的分类类别
            trues = np.argmax(labels, axis=-1)  # 获取该批数据的真实类别
            accuracy = np.mean(np.equal(preds, trues))  # 计算该批数据的准确率
            train_accuracy.append(accuracy)
        optimizer.zero_grad()  # 梯度清零
        model.backward()  # 反向传播
        optimizer.step()  # 更新参数
    # 如果是分类模型返回平均每批数据的损失和准确率作为该轮的损失和准确率，否则只返回损失
    return (np.mean(train_loss), np.mean(train_accuracy)) if classification else np.mean(train_loss)


# 结构与train_model类似
def valid_model(valid_loader, model, classification: bool):
    """模型验证函数"""
    valid_loss = []
    valid_accuracy = [] if classification else None
    for i, data in enumerate(valid_loader):
        inputs, labels = data[0], data[1]
        outputs = model(inputs)  # 前向传播
        loss = model.criterion(outputs, labels)  # 计算该批数据的损失
        valid_loss.append(loss)
        if classification:
            preds = np.argmax(outputs, axis=-1)  # 获取该批数据的分类类别
            trues = np.argmax(labels, axis=-1)  # 获取该批数据的真实类别
            accuracy = np.mean(np.equal(preds, trues))  # 计算该批数据的准确率
            valid_accuracy.append(accuracy)
    # 如果是分类模型返回平均每批数据的损失和准确率作为该轮的损失和准确率，否则只返回损失
    return (np.mean(valid_loss), np.mean(valid_accuracy)) if classification else np.mean(valid_loss)


# 结构与valid_model类似
def test_model(test_loader, model, results_path, classification: bool):
    """模型测试函数"""
    test_loss = []
    test_accuracy = [] if classification else None
    for i, data in enumerate(test_loader):
        inputs, labels = data[0], data[1]
        outputs = model(inputs)  # 前向传播
        loss = model.criterion(outputs, labels)  # 计算该批数据的损失
        test_loss.append(loss)
        if classification:
            preds = np.argmax(outputs, axis=-1)  # 获取该批数据的分类类别
            trues = np.argmax(labels, axis=-1)  # 获取该批数据的真实类别
            accuracy = np.mean(np.equal(pred, true))  # 计算该批数据的准确率
            test_accuracy.append(accuracy)
            with open(results_path, 'a') as f:  # 保存测试集分类结果
                for true, pred in zip(trues, preds):
                    f.write(str(true) + ',' + str(pred) + '\n')
        else:
            with open(results_path, 'a') as f:  # 保存测试集预测结果
                for output in outputs:
                    f.write(str(output) + '\n')
    # 如果是分类模型返回平均每批数据的损失和准确率作为最终的损失和准确率，否则只返回损失
    return (np.mean(test_loss), np.mean(test_accuracy)) if classification else np.mean(test_loss)


# 早停法：当模型在验证集上的得分不再提升时（验证集损失不再下降时）及时停止训练，防止模型过拟合
class EarlyStopping:
    """早停法实现"""
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, log, patience=7, verbose=False, delta=0):
        self.log = log
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, save_path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.log.write(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, save_path):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.log(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        model.save(save_path)
        self.val_loss_min = val_loss


# 日志：将训练过程中的输出提示保存到日志文件中
class Log(object):
    """日志保存"""

    def __init__(self, path):
        self.path = path if path.endswith('.txt') else path + '.txt'

    def write(self, text):
        with open(self.path, 'a') as f:
            f.write(text + '\n')

    def __call__(self, text):
        self.write(text)
