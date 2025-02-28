import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

input = torch.reshape(input,[-1,1,2,2])
print(input.shape)

dataset = torchvision.datasets.CIFAR10("data", train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64,shuffle=False)



class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.BN = nn.BatchNorm2d(num_features=3)
        self.relu1 = nn.ReLU()
        self.sigmoid1 = nn.Sigmoid()
    def forward(self,input):
        input = self.BN(input)
        output = self.relu1(input)
        return output

ynn = Model()
writer = SummaryWriter(log_dir="logs_sigmoid")
i = 1
for data in dataloader:
    imgs, targets = data
    writer.add_images("imgs",imgs, global_step= i)
    output = ynn(imgs)
    writer.add_images("outputs_BN+ReLu",output,global_step=i)
    i = i+1

writer.close()


