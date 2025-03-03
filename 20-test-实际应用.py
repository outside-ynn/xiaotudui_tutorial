import torch
import torchvision
from PIL import Image

image_path = "../imgs/airplane.png"
image = Image.open(image_path)
print(image)
image = image.convert('RGB')

transforms = torchvision.transforms.Compose([torchvision.transforms.Resize([32, 32]),
                                             torchvision.transforms.ToTensor()])
image = transforms(image)

print(image.shape)

from model import *

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

#ynn.load_state_dict("ynn_epoch9.pth"）
ynn.load_state_dict(torch.load("ynn_epoch9.pth"))
print(ynn)
image = torch.reshape(image,[1,3,32,32])


ynn.eval()
with torch.no_grad():
    output = ynn(image)
print(output)

pred = output.argmax(1)
print(pred)
