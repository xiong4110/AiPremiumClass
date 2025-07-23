import torch
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(28*28, 128)
        self.linear2 = nn.Linear(128, 10)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x
    
if __name__ == '__main__':
    model = MyModule()
    x = torch.randn(1, 28*28)
    y = model(x)
    print(y)
