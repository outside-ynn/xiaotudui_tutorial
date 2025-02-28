import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# # 修正输入张量的创建，使用逗号分隔每个子列表
# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float)
# print(input)
# input = torch.reshape(input, [-1,1,5,5])
# print(input.shape)

# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.maxpool = MaxPool2d(3, ceil_mode=True)
#
#     def forward(self, x):
#         x = self.maxpool(x)
#         return x

# ynn = Model()
# output = ynn(input)
#
# print(output)
dataset = torchvision.datasets.CIFAR10("data", train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = MaxPool2d(3, ceil_mode=True)

    def forward(self, x):
        x = self.maxpool(x)
        return x

ynn = Model()
writer = SummaryWriter(log_dir="logs_maxpool")
step = 0
for data in dataloader:
    imgs, targets = data

    output = ynn(imgs)
    writer.add_images("maxpooling", output, global_step=step)
    step = step + 1

writer.close()

# ynn = Model()
# output = ynn(input)
#
# print(output)



