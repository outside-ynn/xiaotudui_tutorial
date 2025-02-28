import torch
import torch.nn.functional as F

# 修正输入张量的创建，使用逗号分隔每个子列表
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

# 修正卷积核张量的创建，使用逗号分隔每个子列表
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])

input = torch.reshape(input, [1,1,5,5])
kernel = torch.reshape(kernel, [1,1,3,3])

print(input.shape)
print(kernel.shape)

output = F.conv2d(input, kernel)
print(output)

output2 = F.conv2d(input, kernel, stride=2)
print(output2)

output3 = F.conv2d(input, kernel, stride=1, padding=1)
print(output3)

