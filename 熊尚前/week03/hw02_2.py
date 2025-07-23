import hw02 as MM
import torch
model = MM.MyModule()
y = model(torch.randn(1, 28*28))
print(y)