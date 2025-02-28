import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./data", train=False, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size = 64)

class Model(nn.Module):
    def __init__(self):
        super().__init__()#super(Model,self).__init__()
        self.conv1 = Conv2d(3,6,3,1,0)

    def forward(self,input):
        input = self.conv1(input)
        return input

ynn = Model()
print(ynn)

writer = SummaryWriter("logs")
step = 1
for data in dataloader:
    imgs, targets =data
    output = ynn(imgs)

    print(imgs.shape)#torch.Size([64, 3, 32, 32])
    print(output.shape)#torch.Size([64, 6, 30, 30])
    writer.add_images("input", imgs, global_step = step)

    output = torch.reshape(output,[-1, 3, 30, 30])
    writer.add_images("output", output, global_step = step)

    step = step + 1







