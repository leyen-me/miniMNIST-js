# miniMNIST-js

这个项目用JavaScript实现了一个**最小**的神经网络，用于分类手写数字，数据集来自 [MNIST dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset?resource=download)。整个实现大约**200行代码**，只使用了标准的JavaScript库。

## 特点

- 两层神经网络（输入层 → 隐藏层 → 输出层）
- 隐藏层使用ReLU激活函数
- 输出层使用Softmax激活函数
- 交叉熵损失函数
- 随机梯度下降（SGD）优化器

## 性能
```bash
Epoch 1, Accuracy: 96.12%, Avg Loss: 0.2188
Epoch 2, Accuracy: 96.98%, Avg Loss: 0.0875
Epoch 3, Accuracy: 97.41%, Avg Loss: 0.0561
Epoch 4, Accuracy: 97.63%, Avg Loss: 0.0383
Epoch 5, Accuracy: 97.63%, Avg Loss: 0.0270
Epoch 6, Accuracy: 97.69%, Avg Loss: 0.0193
Epoch 7, Accuracy: 97.98%, Avg Loss: 0.0143
Epoch 8, Accuracy: 98.03%, Avg Loss: 0.0117
Epoch 9, Accuracy: 98.03%, Avg Loss: 0.0103
Epoch 10, Accuracy: 98.06%, Avg Loss: 0.0094
Epoch 11, Accuracy: 98.06%, Avg Loss: 0.0087
Epoch 12, Accuracy: 98.16%, Avg Loss: 0.0081
Epoch 13, Accuracy: 98.16%, Avg Loss: 0.0078
Epoch 14, Accuracy: 98.18%, Avg Loss: 0.0075
Epoch 15, Accuracy: 98.19%, Avg Loss: 0.0074
Epoch 16, Accuracy: 98.20%, Avg Loss: 0.0072
Epoch 17, Accuracy: 98.24%, Avg Loss: 0.0070
Epoch 18, Accuracy: 98.23%, Avg Loss: 0.0069
Epoch 19, Accuracy: 98.23%, Avg Loss: 0.0069
Epoch 20, Accuracy: 98.22%, Avg Loss: 0.0068
```

## 先决条件

- 无需编译器
- MNIST数据集文件：
  - `train-images.idx3-ubyte`
  - `train-labels.idx1-ubyte`

## 编译

无需编译，直接运行JavaScript代码。

## 使用方法

1. 将MNIST数据集文件放置在`data/`目录中。
2. 运行程序：

   ```bash
   node index.js
   ```

程序将会在MNIST数据集上训练神经网络，并输出每个epoch的准确率和平均损失。

## 配置

您可以在 `index.js` 中调整以下参数：

- `HIDDEN_SIZE`：隐藏层中的神经元数量
- `LEARNING_RATE`：SGD的学习率
- `EPOCHS`：训练的轮数
- `BATCH_SIZE`：训练的小批量大小
- `TRAIN_SPLIT`：用于训练的数据比例（其余用于测试）
