"""
文件：mnist.py
作者：RunRain
日期：2023/1/14
描述：MNIST手写数字识别案例
"""

import sys
import struct
import numpy as np
import time
from tqdm import tqdm

sys.path.append('../..')
sys.path.append('../../pynn')

import pynn.data as Data
from pynn import nn
from pynn import optim
from pynn.net import Net
from pynn.preprocess import *
from pynn.process import *
from pynn.visualization import Graph


class MnistNet(Net):
    def __init__(self, activation, criterion):
        super(MnistNet, self).__init__()
        if activation == 'relu':
            self.struct = nn.Sequential(
                nn.Linear(784, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
            )
        else:  # sigmoid
            self.struct = nn.Sequential(
                nn.Linear(784, 1024),
                nn.Sigmoid(),
                nn.Linear(1024, 512),
                nn.Sigmoid(),
                nn.Linear(512, 256),
                nn.Sigmoid(),
                nn.Linear(256, 128),
                nn.Sigmoid(),
                nn.Linear(128, 64),
                nn.Sigmoid(),
                nn.Linear(64, 10)
            )
        self.criterion = criterion


def load_data(batch_size):
    train_images_path = './dataset/train-images.idx3-ubyte'
    train_labels_path = './dataset/train-labels.idx1-ubyte'
    test_images_path = './dataset/t10k-images.idx3-ubyte'
    test_labels_path = './dataset/t10k-labels.idx1-ubyte'

    with open(train_labels_path, 'rb') as train_labels_file:
        magic, n = struct.unpack('>II', train_labels_file.read(8))
        train_labels = np.fromfile(train_labels_file, dtype=np.uint8)
    with open(train_images_path, 'rb') as train_images_file:
        magic, num, rows, cols = struct.unpack('>IIII', train_images_file.read(16))
        train_images = np.fromfile(train_images_file, dtype=np.uint8).reshape(len(train_labels), 784)
    with open(test_labels_path, 'rb') as test_labels_file:
        magic, n = struct.unpack('>II', test_labels_file.read(8))
        test_labels = np.fromfile(test_labels_file, dtype=np.uint8)
    with open(test_images_path, 'rb') as test_images_file:
        magic, num, rows, cols = struct.unpack('>IIII', test_images_file.read(16))
        test_images = np.fromfile(test_images_file, dtype=np.uint8).reshape(len(test_labels), 784)

    # 标签独热化
    train_onehot_labels = one_hot_encoder(train_labels, 10)  # train_onehot_labels: (60000, 10)
    test_onehot_labels = one_hot_encoder(test_labels, 10)  # test_onehot_labels: (10000, 10)

    # 输入归一化
    train_images = min_max_normalization(train_images)  # train_images: (60000, 784)
    test_images = min_max_normalization(test_images)  # test_images: (10000, 784)

    # 生成数据集 索引返回一个样本(x,y)
    train_set = Data.Dataset(train_images, train_onehot_labels)
    test_set = Data.Dataset(test_images, test_onehot_labels)

    # 将原始训练集随机划分为训练集和验证集
    train_set, valid_set = Data.random_split(train_set, [8, 2])  # 训练集80%，验证集20%

    # 生成data_loader迭代器 迭代得到(batch_X,batch_Y)
    train_loader = Data.DataLoader(train_set, batch_size=batch_size, shuffle=False)  # 上面已经随机划分了，就不用再指定shuffle=True了
    valid_loader = Data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = Data.DataLoader(test_set, batch_size=batch_size, shuffle=False)  # 测试集无需打乱

    return train_loader, valid_loader, test_loader


def start(epochs):
    print('Start Training')
    log('Start Training')
    for epoch in tqdm(range(1, epochs + 1)):
        start_time = time.time()
        train_loss, train_accuracy = train_model(train_loader, model, optimizer, classification=True)  # 训练模型
        train_time = time.time() - start_time
        valid_loss, valid_accuracy = valid_model(valid_loader, model, classification=True)  # 验证模型

        print(
            '[%s] Epoch: %d - Train_Loss: %.6f - Train_Accuracy: %.6f - Valid_Loss: %.6f - Valid_Accuracy: %.6f - Train_Time: %s'
            % (time.strftime('%Y-%m-%d  %H:%M:%S', time.localtime()), epoch, train_loss, train_accuracy, valid_loss,
               valid_accuracy, time.strftime('%H:%M:%S', time.gmtime(train_time))))
        log('[%s] Epoch: %d - Train_Loss: %.6f - Train_Accuracy: %.6f - Valid_Loss: %.6f - Valid_Accuracy: %.6f - Train_Time: %s'
            % (time.strftime('%Y-%m-%d  %H:%M:%S', time.localtime()), epoch, train_loss, train_accuracy, valid_loss,
               valid_accuracy, time.strftime('%H:%M:%S', time.gmtime(train_time))))
        graph(train_loss, train_accuracy, valid_loss, valid_accuracy)  # 绘图
        with open('./loss&accuracy/' + file_name + '.csv', 'a') as f:  # 保存损失和准确率数据到csv文件
            f.write(
                str(train_loss) + ',' + str(train_accuracy) + ',' + str(valid_loss) + ',' + str(valid_accuracy) + '\n')

        early_stopping(valid_loss, model, './weights/' + file_name)  # 判断是否早停，自动保存最优模型
        if early_stopping.early_stop:
            print('Early stopping')
            log('Early stopping')
            break  # 结束训练
    graph.save('./graph/' + file_name + '.png')
    print('Training Finished')
    log('Training Finished')

    print('Start Testing')
    log('Start Testing')
    test_loss, test_accuracy = test_model(test_loader, model, './testset_results/' + file_name + '.csv',
                                          classification=True)  # 测试模型
    print('[%s] - Test_Loss: %.6f - Test_Accuracy: %.6f'
          % (time.strftime('%Y-%m-%d  %H:%M:%S', time.localtime()), test_loss, test_accuracy))
    log('[%s] - Test_Loss: %.6f - Test_Accuracy: %.6f'
        % (time.strftime('%Y-%m-%d  %H:%M:%S', time.localtime()), test_loss, test_accuracy))
    print('Testing Finished')
    log('Testing Finished')


if __name__ == '__main__':
    config = {
        'loss': 'ce',  # 'mse','ce'
        'optimizer': 'adam',  # 'sgd','momentum','adagrad','adam'
        'lr': 1e-3,
        'activation': 'relu',  # 'sigmoid','relu'
        'batch_size': 128,
        'epochs': 30,
        'pretrained': '',  # '','weights_file_name_without_suffix'
    }
    print(str(config))
    file_name = str(int(time.time()))  # Unix时间戳
    print('file_name:', file_name)
    log = Log('./log/' + file_name)
    log(str(config))
    loss = nn.MSELoss() if config['loss'].lower() == 'mse' else nn.CrossEntropyLoss()
    model = MnistNet(config['activation'].lower(), loss)
    model.load('./weights/' + config['pretrained'] + '.npz') if config['pretrained'] else None
    if config['optimizer'].lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'])
    elif config['optimizer'].lower() == 'momentum':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    elif config['optimizer'].lower() == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=config['lr'])
    else:  # adam
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    early_stopping = EarlyStopping(log, patience=5, verbose=True)
    graph = Graph(2, 2, xrange=(0, config['epochs']),
                  titles=('Train_Loss', 'Train_Accuracy', 'Valid_Loss', 'Valid_Accuracy'))  # 图窗2*2
    train_loader, valid_loader, test_loader = load_data(config['batch_size'])
    print('Data Loading Finished')
    start(config['epochs'])
