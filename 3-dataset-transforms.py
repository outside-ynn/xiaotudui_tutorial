import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, transform=dataset_transforms, download=True)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=dataset_transforms, download=True)

# print(test_set[0])
# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes)
# img.show()

print(test_set[0])

writer = SummaryWriter(log_dir="p10")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("P10", img, i)

writer.close()


