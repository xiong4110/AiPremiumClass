import torch 
import torch.nn as nn

dropout = nn.Dropout(0.1)
imputs = torch.randn(10000, dtype=torch.float32)
outputs = dropout(imputs)
print((outputs==0).sum().item())