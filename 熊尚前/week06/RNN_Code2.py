import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class RNN_Classifier(nn.Module):
    def __init__(self):
        # 调用父类 nn.Module 的构造函数，确保正确初始化父类的属性和方法
        super().__init__()
        self.rnn = nn.RNN(
            # 输入特征的维度，这里表示每个时间步的输入特征数量为 28
            input_size=28,
            # 隐藏状态的维度，即 RNN 隐藏层的神经元数量为 50
            hidden_size=50,
            # 是否使用偏置项，True 表示使用
            bias=True,
            # RNN 的层数，这里设置为 5 层
            num_layers=5,
            # 输入和输出张量的第一维是否为批次大小，True 表示是
            batch_first=True,
        )
        # 定义一个全连接层，将 RNN 输出的最后一个时间步的特征映射到 10 个类别上
        self.fc = nn.Linear(50, 10)

    def forward(self, x):
        """
        定义模型的前向传播过程。

        参数:
        x (torch.Tensor): 输入张量，形状通常为 (batch_size, sequence_length, input_size)

        返回:
        torch.Tensor: 经过全连接层处理后的输出张量，形状为 (batch_size, 10)
        """
        # 通过 RNN 层处理输入数据，output 包含所有时间步的输出特征，l_h 是最后一个时间步的隐藏状态
        output, l_h = self.rnn(x)
        # 提取每个样本在最后一个时间步的输出特征，然后通过全连接层进行分类
        out = self.fc(output[:, -1, :])
        # 返回最终的分类结果
        return out


if __name__ == "__main__":

    writer = SummaryWriter()

    # 加载数据集
    train_data = MNIST(root="./data", train=True, transform=ToTensor(), download=True)
    test_data = MNIST(root="./data", train=False, transform=ToTensor(), download=True)

    # 创建数据加载器
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # 实例化模型
    model = RNN_Classifier()

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    EPOCHS = 10
    for epoch in range(EPOCHS):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images.squeeze())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch + 1}/{EPOCHS}], loss: {loss.item()}")
            writer.add_scalar(
                "training loss", loss.item(), epoch * len(train_loader) + i
            )
