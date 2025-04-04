"""
RNN Demo
"""

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

# 创建一个MNIST数据集对象
ds_train = MNIST(root="./data", train=True, download=True, transform=ToTensor())
# 将数据集封装起来，设置数据批量大小为128，并打乱数据顺序
dl_train = DataLoader(ds_train, batch_size=128, shuffle=True)


class MinistClassifier(nn.Module):
    """
    自定义神经网络模型
    """

    def __init__(self, input_size, hidden_size, num_labels):
        """
        :input_size 输入特征维度
        :hidden_size 隐藏层维度
        :batch_first 输入数据的第一个维度是批量大小
        """
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size, hidden_size=hidden_size, batch_first=True
        )
        # 创建一个全连接层，用于将RNN层的输出映射到分类结果
        self.classifier = nn.Linear(in_features=hidden_size, out_features=num_labels)

    def forward(self, input_data):
        output, h_n = self.rnn(
            input_data
        )  # 将输入数据传入RNN层，output是RNN层在每个时间步的输出，h_n是最后一个时间步的隐藏状态
        return self.classifier(
            h_n[0]
        )  # 将最后一个时间步的隐藏状态传入全连接层，得到分类结果


BATCH_SIZE = 16
EPOCHS = 5

model = MinistClassifier(28, 256, 10)
# 优化器、损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    tpbar = tqdm(dl_train)
    for img, lbl in tpbar:
        img = img.squeeze()
        logits = model(img)
        loss = criterion(logits, lbl)
        loss.backward()
        optimizer.step()
        model.zero_grad()
        tpbar.set_description(f"epoch:{epoch+1} train_loss:{loss.item()}")
