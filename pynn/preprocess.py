"""
文件：preprocess.py
作者：RunRain
日期：2023/1/12
描述：数据预处理算法实现
"""

import numpy as np


# labels输入类别列表或ndarray对象，元素取值应为0到class_num-1
def one_hot_encoder(labels: (np.ndarray, list), class_num) -> (np.ndarray, list):
    """独热化"""
    one_hot_labels = []
    for label in labels:
        one_hot_label = [0] * class_num
        one_hot_label[label] = 1
        one_hot_labels.append(one_hot_label)
    if isinstance(labels, np.ndarray):
        return np.array(one_hot_labels)
    else:  # list
        return one_hot_labels


def min_max_normalization(data: (np.ndarray, list)) -> (np.ndarray, list):
    """min-max归一化"""
    normal_data = np.array(data)
    normal_data = (normal_data - np.min(normal_data)) / (np.max(normal_data) - np.min(normal_data))
    if isinstance(data, np.ndarray):
        return normal_data
    else:  # list
        return normal_data.tolist()
