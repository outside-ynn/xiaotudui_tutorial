import torch
import torchvision.models
from torch import nn

vgg16 = torchvision.models.vgg16(weights = None)

#保存方式1,这种方式及不仅保存了模型结构还保存了模型的参数,模型结构+参数
torch.save(vgg16,"vgg16_method1.pth")

#保存方式2，只保存了模型参数
torch.save(vgg16.state_dict(),"vgg16_method2.pth")

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)

    def forward(self, x):
        x = self.conv1(x)
        return x

ynn = model()
torch.save(ynn,"ynn_method1.pth")


