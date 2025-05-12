import numpy as np
import matplotlib.pyplot as plt

# 收集数据，载入数据
def get_data():
    x = np.linspace(-1, 1, 100)
    y = 4*x**4 + 3*x**3 + 2*x**2 + 1*x + 1
    return x, y
    
# 创建模型，假设函数形式
def f(x):
    global W
    return W[0] * x**4 + W[1] * x**3 + W[2] * x**2 + W[3] * x + W[4]
    
# 赋初始值
W = np.ones(5)

# 计算误差
def loss(y, y_h):
    return np.sum((y - y_h)**2)
    
# 计算梯度
def gradient(x, y, y_h):
    grad = np.zeros(5)
    grad[0] = np.sum(-2 * x**4 * (y - y_h))
    grad[1] = np.sum(-2 * x**3 * (y - y_h))
    grad[2] = np.sum(-2 * x**2 * (y - y_h))  
    grad[3] = np.sum(-2 * x * (y - y_h))
    grad[4] = np.sum(-2 * (y - y_h))
    return grad

# 循环迭代
def train(x, y, lr, epochs):
    w_list = []
    loss_list = []
    for epoch in range(epochs):
        y_h = f(x)
        global W
        loss_value = loss(y, y_h)
        grad = gradient(x, y, y_h)
        W = W - lr * grad # 更新参数
        w_list.append(W)
        loss_list.append(loss_value)
    return w_list, loss_list

if __name__ == '__main__':
    x, y = get_data()
    lr = 0.005
    epochs = 50
    w_list, loss_list = train(x, y, lr, epochs)
    print(W)

    plt.plot(loss_list)
    plt.show()

    for i, w in enumerate(w_list):
        W = w
        y_h = f(x)
        plt.cla()
        plt.plot(x, y)
        plt.plot(x, y_h, 'r')
        plt.title(f"{i}")
        plt.draw()
        plt.pause(0.1)