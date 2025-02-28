import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

test_data = torchvision.datasets.CIFAR10(root='./data', train=False, transform=dataset_transforms, download=True)

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

#test_set的第一张图片
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter(log_dir="logs_dataloader")
# step = 0
# for data in test_loader:
#     imgs, targets = data
#     # print(imgs.shape)
#     # print(targets)
#     writer.add_images("logs_dataloader2",imgs,step)
#     step = step + 1

for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch:{}".format(epoch), imgs, step)
        step = step + 1

writer.close()

