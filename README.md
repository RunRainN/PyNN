# PyNN
## 基于Numpy的简单神经网络框架
本项目设计的简单神经网络框架PyNN具备全连接神经网络的训练条件，代码风格类似PyTorch。

### 框架结构图
![PyNN](https://user-images.githubusercontent.com/54665114/212606272-72ce18fa-fd72-4346-9756-0a330ba343b9.png)

### PyNN特点

1. 提供了多种激活函数、损失函数和优化方法，可按如下代码调用：

```python
nn.ReLU()

nn.CrossEntropyLoss()

optim.Adam(model.parameters(), lr=1e-3)
```

2. 实现了网络模型基类，从而可以使用如下方式定义网络：

```python
class FCNN(Net):
    def __init__(self):
        super(FCNN, self).__init__()
        self.struct = nn.Sequential(
            nn.Linear(2, 3),
            # nn.Sigmoid()
            nn.ReLU()
        )
        self.criterion = nn.MSELoss()
 
model = FCNN()
```

3. 基于计算图实现了网络模型的前向传播和反向传播，调用方式如下：

```python
outputs = model(inputs)
model.backward()
```

4. 实现了数据集的生成和划分，以及用于获取批数据的数据加载器，调用方式如下：

```python
# 生成数据集 索引返回一个样本(x,y)
train_set = Data.Dataset(train_images, train_onehot_labels)
# 将原始训练集随机划分为训练集和验证集
train_set, valid_set = Data.random_split(train_set, [8, 2])  # 训练集80%，验证集20%
# 生成data_loader迭代器 迭代得到(batch_X,batch_Y)
train_loader = Data.DataLoader(train_set, batch_size=batch_size, shuffle=False)  # 上面已经随机划分了，就不用再指定shuffle=True了
```

5. 基于Matplotlib实现了训练过程可视化，可在训练中实时更新损失、准确率等指标，调用方式如下：

```python
graph = Graph(2, 2, xrange=(0, epochs), titles=('Train_Loss', 'Train_Accuracy', 'Valid_Loss', 'Valid_Accuracy'))  # 图窗2*2
for epoch in range(1, epochs + 1):
    train_loss, train_accuracy = train_model(train_loader, model, optimizer, classification=True)  # 训练模型
    valid_loss, valid_accuracy = valid_model(valid_loader, model, classification=True)  # 验证模型
    graph(train_loss, train_accuracy, valid_loss, valid_accuracy)  # 绘图
```
效果如下：
![graph](https://user-images.githubusercontent.com/54665114/212611500-cb9d07d7-34d5-4b6b-8acc-a4c597dbad0a.png)

### PyNN文件结构及说明
- PyNN	神经网络框架文件
    - pynn	神经网络框架基础文件
        - nn.py	网络层定义
            - Layer	层基类
            - Parameters	权重参数类
            - Linear	全连接层/线性层/Affine层
            - Sigmoid	Sigmoid激活函数层
            - ReLU	ReLU激活函数层
            - Sequential	层容器
            - MSELoss	均方误差损失函数层
            - CrossEntropyLoss 交叉熵损失函数层（带Softmax）
        - optim.py	优化器定义
            - Optim	优化器基类
            - SGD	带momentum的SGD优化器
            - Adagrad	Adagrad优化器
            - Adam	Adam优化器
        - data.py	数据集及数据加载器定义
            - Dataset	数据集类
            - BatchSampler	批采样器
            - collate_fn	数据组合函数
            - _DataLoaderIter	数据加载器迭代类
            - DataLoader	数据加载器
            - random_split	数据集划分函数
        - net.py	网络模型定义
            - Net	网络模型基类
        - process.py	训练、验证、测试过程相关实现
            - train_model	模型训练函数
            - valid_model	模型验证函数
            - test_model	模型测试函数
            - EarlyStopping	早停法实现
            - Log	日志保存
        - preprocess.py	数据预处理算法实现
            - one_hot_encoder	独热化
            - min_max_normalization	min-max归一化
        - functional.py	运算函数实现
            - sigmoid	Sigmoid函数
            - relu	ReLU函数
            - softmax	Softmax函数
        - visualization.py	训练过程可视化实现
            - Graph	图窗类
    - examples	案例文件
        - test_fcnn.py	FCNN测试
        - mnist	MNIST手写数字识别
            - dataset	数据集
            - log	程序运行日志
            - loss&accuracy	损失和准确率数据
            - test_results	测试集识别结果
            - weights	模型权重参数
            - mnist.py	主程序
            - test_table.xlsx	试验表格
