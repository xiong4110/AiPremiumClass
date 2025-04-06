import torch
import torch.nn 

rnn = torch.nn.RNN(
    input_size = 28,
    hidden_size = 50,
    bias = True,
    batch_first = True
)

X = torch.randn(10, 28, 28)

outputs, l_h = rnn(X)

print(outputs.shape)
print(l_h.shape)

