
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("data",train=False,transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset,batch_size=64)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024,10)#注意检查网络的正确性，这里可以在最后面检查output的shape，如果你不知道这一行flatten过后是多少，也可以把这一行注释掉，最后输出shape

        )


    def forward(self,x):
        x = self.model1(x)
        return x

ynn = Model()
print(ynn)

# x = torch.ones([32,3,32,32])
# output = ynn(x)
# print(output.shape)

loss = nn.CrossEntropyLoss()

for data in dataloader:
    imgs, tags = data
    outputs = ynn(imgs)
    # print(outputs)
    # print(tags)
    result_loss = loss(outputs,tags)
    print(result_loss)
    result_loss.backward()#这个backward是非常重要的关乎到有没有grad7
    print("ok")























