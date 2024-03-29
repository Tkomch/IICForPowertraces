import torchvision
from torch import nn
import sys
sys.path.append('..')
from config import *

class DoubleBatchTransform(nn.Module):
    def __init__(self):
        super(DoubleBatchTransform, self).__init__()
        # 定义灰度变换
        self.grey = torchvision.transforms.Grayscale()
        # 定义随机旋转
        self.rotate = torchvision.transforms.RandomRotation(180)
        # 定义水平翻转
        self.flip_hor = torchvision.transforms.RandomHorizontalFlip()
        # 定义垂直翻转
        self.flip_ver = torchvision.transforms.RandomVerticalFlip()
        # 定义随机尺寸缩放
        self.resize = torchvision.transforms.RandomResizedCrop(resize_shape_x, scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333))
        # 定义随机亮度变换
        self.brightness = torchvision.transforms.ColorJitter(brightness=0.5)
        # 定义随机对比度变换
        self.contrast = torchvision.transforms.ColorJitter(contrast=0.5)
        # 定义随机饱和度变换
        self.saturation = torchvision.transforms.ColorJitter(saturation=0.5)
        # 定义随机色相变换
        self.hue = torchvision.transforms.ColorJitter(hue=0.5)
        # 定义随机高斯模糊
        self.gaussian_blur = torchvision.transforms.GaussianBlur(5)
        # 定义标准化
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, x):
        x1 = self.grey(x)
        # 灰度化后对通道进行扩展
        x1 = x1.expand(-1, 3, -1, -1)
        x1 = self.rotate(x1)
        x1 = self.flip_hor(x1)
        x2 = self.resize(x)
        x2 = self.flip_ver(x2)
        x2 = self.normalize(x2)
        return x1, x2
        
