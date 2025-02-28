import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d
from torch.utils.tensorboard import SummaryWriter


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = nn.Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,1, padding=2),
            MaxPool2d(2),
            Conv2d(32,64,5,1,2),
            MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024,64),
            nn.Linear(64,10)
        )

    def forward(self, input):
        output = self.model1(input)
        return output

ynn = Model()
print(ynn)

input = torch.ones([64,3,32,32])
output = ynn(input)
print(output.shape)

writer = SummaryWriter(log_dir="logs_vgg")
writer.add_graph(ynn, input)
writer.close()