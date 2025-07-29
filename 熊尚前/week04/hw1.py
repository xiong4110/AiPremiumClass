# 搭建的神经网络，使用olivettiface数据集进行训练。

# Olivetti Faces 数据集是一个经典的人脸图像数据集。
# 常用于计算机视觉和机器学习中的人脸识别、人脸重建、特征提取等任务。
# 包含 40 位不同个体的人脸图像，64 * 64像素的灰度图像，每位个体有 10 张样本图像，总计 400 张图像。
#


from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn


def get_data():
    """
    加载 Olivetti Faces 数据集，并将其划分为训练集和测试集，最后转换为 PyTorch 张量。

    Returns:
        X_train (torch.Tensor): 训练集特征，数据类型为 torch.float32。
        y_train (torch.Tensor): 训练集标签，数据类型为 torch.long。
        X_test (torch.Tensor): 测试集特征，数据类型为 torch.float32。
        y_test (torch.Tensor): 测试集标签，数据类型为 torch.long。
    """
    # 从指定路径加载 Olivetti Faces 数据集
    # data_home 参数指定数据集的存储路径为 "./face_data"
    olivetti_faces = fetch_olivetti_faces(data_home="./face_data")
    # 提取数据集中的图像数据，存储在 images 变量中
    images = olivetti_faces.data
    # 提取数据集中的目标标签，存储在 targets 变量中
    targets = olivetti_faces.target

    # 使用 train_test_split 函数将数据集划分为训练集和测试集
    # test_size=0.2 表示测试集占总数据集的 20%
    # random_state=42 用于固定随机种子，确保每次划分结果相同
    X_train, X_test, y_train, y_test = train_test_split(
        images, targets, test_size=0.2, random_state=42
    )

    # 将训练集和测试集的特征和标签转换为 PyTorch 张量
    # dtype=torch.float32 指定特征数据类型为 32 位浮点数
    # dtype=torch.long 指定标签数据类型为 64 位整数
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    return X_train, y_train, X_test, y_test

# 1.定义模型，普通模型
model0 = nn.Sequential(
    nn.Linear(64 * 64, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 40),
)

# 2.定义模型，结合归一化，正则化
model1 = nn.Sequential(
    nn.BatchNorm1d(64 * 64),
    nn.Linear(64 * 64, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 40),
)
# 定义损失函数
loss = nn.CrossEntropyLoss()

# 定义优化器
def self_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=0.001)


def train_model(X, y, model, optimizer, loss):
    """
    训练给定的 PyTorch 模型。

    Args:
        X (torch.Tensor): 输入特征张量，用于模型训练。
        y (torch.Tensor): 目标标签张量，用于计算损失。
        model (torch.nn.Module): 待训练的 PyTorch 模型。
        optimizer (torch.optim.Optimizer): 用于更新模型参数的优化器。
        loss (torch.nn.modules.loss._Loss): 用于计算损失的损失函数。

    Returns:
        list: 包含每一轮训练损失值的列表。
    """
    loss_values = []
    # 训练模型，共进行 100 轮迭代
    for i in range(100):
        # 前向传播：将输入数据 X 传入模型，得到预测结果
        y_pred = model(X)
        # 计算损失：使用指定的损失函数计算预测结果 y_pred 与真实标签 y 之间的损失
        loss_value = loss(y_pred, y)
        # 将当前轮次的损失值添加到损失值列表中
        loss_values.append(loss_value.item())
        # 反向传播前，将优化器中的梯度缓存清零，防止梯度累积
        optimizer.zero_grad()
        # 反向传播：计算损失函数关于模型参数的梯度
        loss_value.backward()
        # 更新参数：使用优化器根据计算得到的梯度更新模型的参数
        optimizer.step()
    return loss_values

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_data()
    loss_values0 = train_model(X_train, y_train, model0, self_optimizer(model0), loss)
    loss_values1 = train_model(X_train, y_train, model1, self_optimizer(model1), loss)
    # 绘制损失曲线
    plt.plot(loss_values0, label="model0")
    plt.plot(loss_values1, label="model1")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss")
    plt.show()