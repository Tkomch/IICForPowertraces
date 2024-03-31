import sys
import os
import torch
import torch.nn as nn
# 可视化张量图片
from DataTransers.TraceTranser import TripleBatchTransform
from DataLoaders.LoadPartASCAD import Datasetloader
from config import *
# 引入网络结构
from Nets.cnn_single_head_3layer import CNNNet
from Nets.Resnet import ResNet_18
from utils import setup_seed
from tqdm import tqdm

setup_seed(seed)

def train():
    train_loader, _ = Datasetloader(train_data_path)(bs, is_shuffle, dataset_mode, left, right)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    loss_list = []
    tran = TripleBatchTransform()
    # 更改网络结构在这里
    print("将使用%s网络结构进行训练" % net_structure)
    if (net_structure == 'cs3'):
        from Nets.cnn_single_head_3layer import CNNNet
        model = CNNNet().to(device)
    elif (net_structure == 'resnet18'):
        from Nets.Resnet import ResNet_18
        model = ResNet_18().to(device)
    elif (net_structure == 'cs4'):
        from Nets.cnn_single_head import CNNNet
        model = CNNNet().to(device)
    elif (net_structure == 'ms5'):
        from Nets.mlp_5layer import MLPNet
        model = MLPNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print(f"epoch: {epoch}")
        for i, data in enumerate(tqdm(train_loader), 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            inputs_trans = tran(inputs)
            optimizer.zero_grad()
            outputs = model(inputs_trans)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        loss_list.append(running_loss / len(train_loader))
        print(f"Loss: {running_loss / len(train_loader)}")
        if (saveModel == True):
            if not os.path.exists(saveModelPath):
                os.makedirs(saveModelPath)
            torch.save(model, saveModelPath + f"model_{epoch}.pth")
    # 保存模型
    print(f"最终模型保存在{modelsaveName}")
    torch.save(model, modelsaveName)
    with open(lossSavePath, "w") as f:
        for loss_epoch in loss_list:
            f.write(str(loss_epoch) + "\n")
    
train()
