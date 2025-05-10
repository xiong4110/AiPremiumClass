import torch
import torch.nn as nn
from sklearn.datasets import fetch_olivetti_faces
from torch.utils.tensorboard import SummaryWriter


class RNN_Demo(nn.Module):

    def __init__(self, rnn_type="LSTM"):
        super().__init__()
        self.rnn_type = rnn_type
        self.init_rnn(rnn_type)
        self.fc = nn.Linear(128, 40)  # 假设olivetti人脸数据集有40个人

    def init_rnn(self, rnn_type="LSTM"):
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=64,  # 输入特征维度
                hidden_size=128,  # 隐藏层维度
                bias=True,  # 是否使用偏置
                num_layers=2,  # 层数
                batch_first=True,  # 输入数据的第一个维度是batch_size
            )
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=64,  # 输入特征维度
                hidden_size=128,  # 隐藏层维度
                bias=True,  # 是否使用偏置
                num_layers=2,  # 层数
                batch_first=True,  # 输入数据的第一个维度是batch_size
            )
        elif rnn_type == "RNN":
            self.rnn = nn.RNN(
                input_size=64,  # 输入特征维度
                hidden_size=128,  # 隐藏层维度
                bias=True,  # 是否使用偏置
                num_layers=2,  # 层数
                batch_first=True,  # 输入数据的第一个维度是batch_size
            )
        else:
            raise ValueError("Unsupported RNN type")

    def forward(self, x):
        outputs, l_h = self.rnn(x)
        return self.fc(outputs[:, -1, :])  # 取最后一个时间步的输出


if __name__ == "__main__":
    writer = SummaryWriter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 下载olivetti人脸数据集
    olivetti_faces = fetch_olivetti_faces(data_home="./olivetti_faces")
    
    # 读取数据
    data = torch.tensor(olivetti_faces.images, dtype=torch.float32)
    labels = torch.tensor(olivetti_faces.target, dtype=torch.long)
    
    idx = torch.randperm(data.size(0))  # 生成
    data = data[idx]
    labels = labels[idx]
    # 划分训练集和测试集
    train_data = data[:300]
    train_labels = labels[:300]
    test_data = data[300:]
    test_labels = labels[300:]
    
    # 构建数据集
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)
    
    # 构建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False
    )

    def build_rnn(rnn_type="LSTM"):
        model = RNN_Demo(rnn_type).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        return model, criterion, optimizer

    rnn_types = ["LSTM", "GRU", "RNN"]
    for rnn_type in rnn_types:
        model, criterion, optimizer = build_rnn(rnn_type)
        print(f"Training {rnn_type} model...")
        # 训练模型
        epochs = 200
        for epoch in range(epochs):
            model.train()
            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images.squeeze())

                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                if i % 100 == 0:
                    print(
                        "{}: Epoch [{}/{}], Loss: {:.4f}".format(
                            rnn_type, epoch + 1, epochs, loss.item()
                        )
                    )
                    writer.add_scalar(
                        f"{rnn_type}: training loss",
                        loss.item(),
                        epoch * len(train_loader) + i,
                    )

            # 评估模型
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images.squeeze())
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                accuracy = 100 * correct / total
                print(
                    "{}: Epoch [{}/{}], Accuracy: {:.2f}%".format(
                        rnn_type, epoch + 1, epochs, accuracy
                    )
                )
                writer.add_scalar(
                    f"{rnn_type}: testing accuracy", accuracy, epoch
                )
