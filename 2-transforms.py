from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

image_path =r"E:\tudui-pytorch-tutorial\imgs\002.jpg"
img = Image.open(image_path)
print(img)

writer = SummaryWriter('logs')
tensor_trans = transforms.ToTensor()#这个实际上是类
tensor_img =tensor_trans(img)#这里是将这个类实例化，所以会调用类中的call函数，这里的img是call所需要的参数

writer.add_image("tensor", tensor_img, 0)

writer.close()
print(tensor_img)
