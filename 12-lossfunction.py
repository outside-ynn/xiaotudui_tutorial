import torch

from torch import nn
from torch.nn import L1Loss

input = torch.tensor([1,2,3], dtype=torch.float32)
targets = torch.tensor([1,2,5], dtype=torch.float32)

input = torch.reshape(input,[1,1,1,3])
targets = torch.reshape(targets,[1,1,1,3])

loss = L1Loss()
result = loss(input,targets)
print(result)

loss = nn.MSELoss()
result2 = loss(input, targets)
print(result2)

x = torch.tensor([0.1,0.2,0.3])
y = torch.tensor([1])
x = torch.reshape(x,[1,3])

'''
x 需要 reshape 的原因：代码中初始的 x = torch.tensor([0.1, 0.2, 0.3]) ，它是一个一维张量。但根据 nn.CrossEntropyLoss 的要求，
它应该表示每个样本对不同类别的预测得分，因此需要将其形状调整为 (N, C) 的形式。这里通过 torch.reshape(x,[1,3]) 将其变为 (1, 3) 的形状，
表示一个样本（批量大小为 1 ）对 3 个类别（C = 3 ）的预测得分。
y 只有一个维度的原因：y = torch.tensor([1]) 只有一个维度，因为它表示的是样本对应的真实类别索引。在这个例子中，批量大小为 1，所以只有一个值 1 ，
表示这个样本对应的真实类别是索引为 1 的类别（假设类别从 0 开始计数） ，符合 nn.CrossEntropyLoss 对包含类别索引的 y 的形状要求。
简单来说，nn.CrossEntropyLoss 对输入的 x 和 y 形状有特定规范，代码中的 reshape 操作和 y 的维度设置都是为了满足这个规范，以正确计算交叉熵损失。
'''

loss_cross = nn.CrossEntropyLoss()
result3 = loss_cross(x, y)
print(result3)

