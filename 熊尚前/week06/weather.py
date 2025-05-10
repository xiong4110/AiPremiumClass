import csv
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 

def get_data():
    # 所有气象站最高气温数据
    stations_maxtemps = {}
    with open('Weather/Summary of Weather.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sta = row['STA']
            stations_maxtemps[sta] = stations_maxtemps.get(sta, []) 
            stations_maxtemps[sta].append(float(row['MaxTemp']))

        print('总共{}个气象站'.format(len(stations_maxtemps)))

        # 过滤掉长度过短的样本
        max_temps = [temps for temps in stations_maxtemps.values() if len(temps) > 20]

        # 过滤掉异常值
        filted_maxtemps = [[temp for temp in temps if temp > -17] for temps in max_temps]

    return filted_maxtemps


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y=None, train=True):
        self.X = X
        self.y = y
        self.train = train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if self.train:
            return self.X[index], self.y[index]
        else:
            return self.X[index]


def generate_time_series(temp_values, batch_size, n_steps):
    # 初始化一个全零的数组，用于存储生成的时间序列数据，形状为 (batch_size, n_steps)
    series = np.zeros((batch_size, n_steps))
    # 获取气象站数据的数量
    sta_size = len(temp_values)
    # 随机生成 batch_size 个整数，范围在 0 到 sta_size 之间，用于随机选择气象站
    sta_idx = np.random.randint(0, sta_size, batch_size)

    # 遍历每个随机选择的气象站索引
    for i, idx in enumerate(sta_idx):
        # 根据索引获取对应气象站的最高气温数据
        temps = temp_values[idx]
        # 获取该气象站最高气温数据的长度
        temp_size = len(temps)
        
        # 随机生成一个整数，范围在 0 到 temp_size - n_steps 之间，用于确定时间序列的起始位置
        random_index = np.random.randint(0, temp_size - n_steps)
        # 从选定的气象站数据中截取长度为 n_steps 的时间序列，并赋值给 series 数组的第 i 行
        series[i] = np.array(temps[random_index:random_index + n_steps])
    # 返回生成的时间序列数据，形状为 (batch_size, n_steps, 1)，以及对应的目标值，形状为 (batch_size, 1)
    return series[:,:n_steps,np.newaxis].astype(np.float32), series[:,-1,np.newaxis].astype(np.float32)


n_steps = 21
max_temps = get_data()

X_train, y_train = generate_time_series(max_temps, 7000, n_steps)
X_valid, y_valid = generate_time_series(max_temps, 2000, n_steps)
X_test, y_test = generate_time_series(max_temps, 1000, n_steps)

dataset = {
    'train': TimeSeriesDataset(X_train, y_train),
    'valid': TimeSeriesDataset(X_valid, y_valid),
    'test': TimeSeriesDataset(X_test, train=False)
}

dataloader = {
    'train': DataLoader(dataset['train'], batch_size=64, shuffle=True),
    'valid': DataLoader(dataset['valid'], batch_size=64, shuffle=False), 
    'test': DataLoader(dataset['test'], batch_size=64, shuffle=False)
}

class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x, h = self.rnn(x)
        y = self.fc(x[:,-1])
        return y
    
device = "cuda" if torch.cuda.is_available() else "cpu"
def fit(model, dataloader, epochs=10, lr=0.001):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    bar = tqdm(range(1, epochs+1))
    for epoch in bar:
        model.train()
        train_loss = []
        for batch in dataloader['train']:
            X, y = batch
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        model.eval()
        val_loss = []
        with torch.no_grad():
            for batch in dataloader['valid']:
                X, y = batch
                X, y = X.to(device), y.to(device)
                y_hat = model(X)
                loss = criterion(y_hat, y)
                val_loss.append(loss.item())
        bar.set_description(f'loss {np.mean(train_loss):.5f}, val_loss {np.mean(val_loss):.5f}' )

def predict(model, dataloader):
    model.eval()
    with torch.no_grad():
        preds = torch.tensor([]).to(device)
        for batch in dataloader:
            X = batch
            X = X.to(device)
            pred = model(X)
            preds = torch.cat([preds, pred])
        return preds
    

def plot_series(series, y=None, y_pred=None, y_pred_std=None, x_label="$t$", y_label="$x$", save_path='./result.png'):
    r, c = 3, 5
    fig, axes = plt.subplots(nrows=r, ncols=c, sharey=True, sharex=True, figsize=(20, 10))
    for row in range(r):
        for col in range(c):
            plt.sca(axes[row][col])
            ix = col + row*c
            plt.plot(series[ix, :], ".-")
            if y is not None:
                plt.plot(range(len(series[ix, :]), len(series[ix, :])+len(y[ix])), y[ix], "bx", markersize=10)
            if y_pred is not None:
                plt.plot(range(len(series[ix, :]), len(series[ix, :])+len(y_pred[ix])), y_pred[ix], "ro")
            if y_pred_std is not None:
                plt.plot(range(len(series[ix, :]), len(series[ix, :])+len(y_pred[ix])), y_pred[ix] + y_pred_std[ix])
                plt.plot(range(len(series[ix, :]), len(series[ix, :])+len(y_pred[ix])), y_pred[ix] - y_pred_std[ix])
            plt.grid(True)
            # plt.hlines(0, 0, 100, linewidth=1)
            # plt.axis([0, len(series[ix, :])+len(y[ix]), -1, 1])
            if x_label and row == r - 1:
              plt.xlabel(x_label, fontsize=16)
            if y_label and col == 0:
              plt.ylabel(y_label, fontsize=16, rotation=0)
    plt.savefig(save_path)
    plt.show()

if __name__ == '__main__':
    fit(RNN(),dataloader, epochs=50)
    y_pred = predict(RNN(), dataloader['test'])
    plot_series(X_test, y_test, y_pred.cpu().numpy())  # cpu() 张量值从gpu搬运到计算机内存
    mean_squared_error(y_test, y_pred.cpu())