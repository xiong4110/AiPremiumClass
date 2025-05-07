# 导入 PyTorch 库，PyTorch 是一个用于深度学习的开源机器学习库
import torch
# 导入 PyTorch 的神经网络模块，该模块包含了构建神经网络所需的各种组件
import torch.nn 

# 创建一个循环神经网络（RNN）实例
# input_size: 输入特征的维度，这里设置为 28，表示每个时间步的输入特征数量为 28
# hidden_size: 隐藏状态的维度，这里设置为 50，表示 RNN 隐藏层的神经元数量为 50
# bias: 是否使用偏置项，True 表示使用，默认值为 True
# batch_first: 若为 True，则输入和输出张量的第一维为批量大小，这里设置为 True
rnn = torch.nn.RNN(
    input_size = 28,
    hidden_size = 50,
    bias = True,
    batch_first = True
)

# 生成一个随机输入张量 X
# 张量的形状为 (10, 28, 28)，其中第一维 10 表示批量大小，即一次处理 10 个样本
# 第二维 28 表示时间步的数量，即每个样本包含 28 个时间步
# 第三维 28 表示每个时间步的输入特征数量，与 RNN 的 input_size 对应
X = torch.randn(10, 28, 28)

# 将输入张量 X 传入 RNN 模型进行前向传播
# outputs: 包含每个时间步的输出特征，形状为 (批量大小, 时间步数量, 隐藏状态维度)
# l_h: 最后一个时间步的隐藏状态，形状为 (1, 批量大小, 隐藏状态维度)
outputs, l_h = rnn(X)

# 打印输出张量 outputs 的形状
print(outputs.shape)
# 打印最后一个时间步的隐藏状态 l_h 的形状
print(l_h.shape)

