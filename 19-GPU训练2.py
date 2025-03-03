import torch
import torchvision.datasets
from torch.utils.data import DataLoader
# from model import *
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import time

# 准备数据集
train_data = torchvision.datasets.CIFAR10("data", train=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10("data", train=False, transform=torchvision.transforms.ToTensor())

train_data_len = len(train_data)
test_data_len = len(test_data)
print("训练数据集的长度为:{}".format(train_data_len))
print("测试数据集的长度为:{}".format(test_data_len))

'''直接在这里定义device'''
device = torch.device("cuda")

# dataloader
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

'''一般来讲都是单开一个model来创建网络模型'''


# model = Modle()
# print(model)
#
# # x = torch.ones([64,3,32,32])#这里是为了计算这个的flatten过后的torch.size
# # output = model(x)
# # print(output.shape)


# 创建网络模型

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 10)

        )

    def forward(self, x):
        x = self.model(x)
        return x

ynn = Model()
ynn = ynn.to(device)
# if torch.cuda.is_available():
#     ynn = ynn.cuda()

# 损失函数loss_fn
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
# if torch.cuda.is_available():
#     loss_fn = loss_fn.cuda()

# 优化器 optimizer
learning_rate = 1e-2
optimizer = torch.optim.SGD(ynn.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练轮数
epoch = 10

writer = SummaryWriter("logs_train")
start_time = time.time()

for i in range(epoch):
    print("---------第{}轮训练开始----------".format(i + 1))
    # 训练步骤开始
    # nn.train,torch.train这都是不对的
    ynn.train()
    for data in train_dataloader:
        # optimizer.zero_grad()#可以写在这里嘛
        imgs, targets = data

        imgs = imgs.to(device)
        targets = targets.to(device)
        # if torch.cuda.is_available():
        #     imgs = imgs.cuda()
        #     targets = targets.cuda()

        outputs = ynn(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        # 由于测试数据总是被掩盖，所以这里逢一百打印一次
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数：{},loss:{}".format(total_train_step, loss.item()))  # .item()会转化成数字
            writer.add_scalar("train_scalar", loss.item(), global_step=total_train_step)

    # 测试步骤开始
    ynn.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():  # 这里的括号不要忘记了
        for data in test_dataloader:
            imgs, targets = data  # 绿色下划线是说明英文单词的拼写错误，英文里面没有这个单词（友情提示）

            imgs = imgs.to(device)
            targets = targets.to(device)

            # if torch.cuda.is_available():
            #     imgs = imgs.cuda()
            #     targets = targets.cuda()

            outputs = ynn(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            preds = torch.argmax(outputs, 1)
            accuracy = torch.sum(preds == targets) #黄色下划线是说不符合语法规范
            total_accuracy += accuracy.item()

    print("整体测试集的正确率：{}".format(total_accuracy / test_data_len))
    print("整体测试集上的loss：{}".format(total_test_loss))
    writer.add_scalar("accuracy", total_accuracy, total_test_step)
    writer.add_scalar("test_scalar", total_test_loss, global_step=total_test_step)
    total_test_step = total_test_step + 1

    torch.save(ynn.state_dict(), "ynn_epoch{}.pth".format(i))#torch.save(ynn, f"ynn_epoch_{i}.pth")
    '''所以这里你还不能用汉字来保存这个模型的权重'''
    print("模型已经保存")

writer.close()
