"""
文件：data.py
作者：RunRain
日期：2023/1/12
描述：数据集及数据加载器定义
"""

import numpy as np


class Dataset(object):
    """数据集类"""

    def __init__(self, X: (list, np.ndarray), Y: (list, np.ndarray)):  # 传入矩阵时数据类型可以为[[]],nd(2d),[nd(1d)]
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)


class BatchSampler(object):
    """批采样器"""

    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:  # batch内索引个数达到batch_size则返回batch
                yield batch
                batch = []  # 下一次调用清空batch，重新从sampler取索引存入batch
        # 此处若判断len(batch)==0则说明上一次返回的batch为最后一批，刚好为batch_size个，之后会抛出异常使外部for循环结束
        if len(batch) > 0 and not self.drop_last:
            yield batch  # 这里返回的是不满batch_size个的batch，为最后一批。下一次调用会抛出异常使外部for循环结束

    def __len__(self):  # 计算批数
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


# 从样本集合中提取输入和输出组合成输入矩阵和输出矩阵（或向量）
def collate_fn(batch_set):
    """数据组合函数"""
    batch_size = len(batch_set)
    inputs = np.stack([item[0] for item in batch_set])
    labels = np.stack([item[1] for item in batch_set])
    return inputs, labels


class _DataLoaderIter:
    """数据加载器迭代类"""

    def __init__(self, loader):
        self.loader = loader
        self.sample_iter = iter(self.loader.batch_sampler)  # 获取batch_sampler对象的迭代器

    def __next__(self):
        index = next(self.sample_iter)
        batch = collate_fn([self.loader.dataset[idx] for idx in index])
        return batch


class DataLoader(object):
    """数据加载器"""

    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        if shuffle:
            self.sampler = np.random.permutation(len(dataset)).tolist()  # 0到总样本数量-1的乱序排列的列表
        else:
            self.sampler = range(len(dataset))  # 0到总样本数量-1的顺序排列的列表
        self.batch_sampler = BatchSampler(self.sampler, batch_size, drop_last)

    def __iter__(self):
        return _DataLoaderIter(self)  # 为支持多轮训练而设计：在多轮训练中，for循环会多次调用iter(dataloader)

    def __len__(self):
        return len(self.batch_sampler)


# dataset = Dataset([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# data_loader = DataLoader(dataset, 2, True)
# for j in range(2):
#     for i in data_loader:
#         print(i[0], 'x')
#         print(i[1], 'y')
#     print(len(data_loader))


def random_split(dataset: Dataset, ratio: (list, tuple)):
    """数据集划分函数"""
    # 例如：len(dataset)==36，ratio为[7,2,1]
    total = len(dataset)
    index = np.random.permutation(total)  # 打乱原始数据集索引
    ratio = np.array(ratio) / np.sum(ratio)  # 转换成百分比小数
    part_number = []
    for i in range(len(ratio) - 1):
        part_number.append(int(total * ratio[i]))
    part_number.append(total - sum(part_number))  # 不遗漏样本 part_number为[25 7 4]
    accum = np.cumsum(part_number)  # 累加的值与各部分索引有关 accum为[25 32 36]，各部分索引为[0:25],[25:32],[32:36]
    part_dataset = []
    last_idx = 0
    for idx in accum:
        part_index = index[last_idx:idx]  # 从打乱的索引中取出各部分索引
        part_X = [dataset[i][0] for i in part_index]
        part_Y = [dataset[i][1] for i in part_index]
        part_dataset.append(Dataset(part_X, part_Y))
        last_idx = idx
    return part_dataset

# dataset = Dataset(range(36), range(35, -1, -1))
# a, b, c = random_split(dataset, [7, 2, 1])
# print(len(a), len(b), len(c))
# for i in range(len(a)):
#     print(a[i])
# print('---------------')
# for i in range(len(b)):
#     print(b[i])
# print('---------------')
# for i in range(len(c)):
#     print(c[i])
