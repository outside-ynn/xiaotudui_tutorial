# import torch
# from torch import nn
# import torchvision
# from torch.utils.data import DataLoader
#
# dataset = torchvision.datasets.CIFAR10("data",train=False,transform=torchvision.transforms.ToTensor())
#
# dataloader = DataLoader(dataset, batch_size=64)
#
# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear1 = nn.Linear(64*3*32*32,10)
#
#     def forward(self,input):
#         output = self.linear1(input)
#         return output
#
# ynn = Model()
#
# for data in dataloader:
#     imgs, targets =data
#     print(imgs.shape)
#     output = torch.reshape(imgs,[1,1,1,-1])
#     print(output.shape)
#     output = ynn(output)
#     print(output.shape)


import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("data",train=False,transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64*3*32*32,10)

    def forward(self,input):
        output = self.linear1(input)
        return output

ynn = Model()

for data in dataloader:
    imgs, targets =data
    print(imgs.shape)
    output = torch.flatten(imgs)
    print(output.shape)
    output = ynn(output)
    print(output.shape)
