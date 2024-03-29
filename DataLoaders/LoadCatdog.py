import os
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
import sys
sys.path.append('..')
from config import *

class Datasetloader():
    def __init__(self, train_data_path, bs, is_shuffle, resize_shape_x, resize_shape_y):
        self.check_file_exists(train_data_path)
        # 定义数据预处理
        transform = transforms.Compose([
            transforms.Resize((resize_shape_x, resize_shape_y)),  # 图像尺寸调整
            transforms.ToTensor(),  # 转换为张量
        ])
        # 加载数据集
        train_dataset = torchvision.datasets.ImageFolder(root=train_data_path, transform=transform)
        # 定义数据加载器
        self.train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=is_shuffle)

    def __call__(self):
        return self.train_loader

    def check_file_exists(self, file_path):
        if os.path.exists(file_path) == False:
            print("Error: provided file path '%s' does not exist!" % file_path)
            sys.exit(-1)
        return


