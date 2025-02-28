import torch
from torch import nn, tensor



class Model(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, input):
        output = input + 1
        return output


model = Model()

x = torch.tensor(12)
out = model(x)
print(out)
