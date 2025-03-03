import torch.nn
import torchvision.datasets
from torch.utils.data import DataLoader
from model import *
from torch import nn

#准备数据集
train_data = torchvision.datasets.CIFAR10("data",train=True,transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10("data",train=False,transform=torchvision.transforms.ToTensor())

train_data_len = len(train_data)
test_data_len = len(test_data)
print("训练数据集的长度为:{}".format(train_data_len))
print("测试数据集的长度为:{}".format(test_data_len))

#dataloader
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)


'''一般来讲都是单开一个model来创建网络模型'''
# class Modle(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = torch.nn.Sequential(
#             nn.Conv2d(3,32,5,1,2),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32,32,5,1,2),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32,64,5,1,2),
#             nn.MaxPool2d(2),
#             nn.Flatten(),
#             nn.Linear(1024,10)
#
#         )
#
#     def forward(self,x):
#         x = self.model(x)
#         return x
#
# model = Modle()
# print(model)
#
# # x = torch.ones([64,3,32,32])#这里是为了计算这个的flatten过后的torch.size
# # output = model(x)
# # print(output.shape)


#创建网络模型
ynn = Model()


#损失函数loss_fn
loss_fn = nn.CrossEntropyLoss()

#优化器 optimizer
learning_rate = 1e-2
optimizer = torch.optim.SGD(ynn.parameters(),lr=learning_rate)

#设置训练网络的一些参数
#记录训练的次数
total_train_step = 0
#记录测试的次数
total_test_step = 0
#训练轮数
epoch = 10

for i in range(epoch):
    print("--------第{}轮训练开始----------".format(i))
    #训练步骤开始
    for data in train_dataloader:
        # optimizer.zero_grad()#可以写在这里嘛
        imgs, targets = data
        outputs = ynn(imgs)
        loss = loss_fn(outputs,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        print("训练次数：{},loss:{}".format(total_train_step, loss.item()))#.item()会转化成数字









