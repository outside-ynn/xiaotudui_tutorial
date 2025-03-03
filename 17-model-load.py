import torch
import torchvision
from model import *
# #方式1加载模型
# model1 = torch.load("vgg16_method1.pth")
# print(model1)
#
# # 方式2加载模型
# vgg16 = torchvision.models.vgg16(weights = None)#方式2需要先构建模型架构
# vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model2_dict = torch.load("vgg16_method2.pth")#这里出来的是一个字典的形式并非一个model
# print(vgg16)

#方式1的陷阱
model1= torch.load('model_instance.pth')#这种方式你必须要反序列化得到哪个model类在哪
print(model1)