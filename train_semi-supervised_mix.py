import sys
import os
import torch
import torch.nn as nn
# 可视化张量图片
# from torchvision.transforms import ToPILImage
# from DataTransers.ImageTranser import DoubleBatchTransform
from DataTransers.TraceTranser import TripleBatchTransform
from IIC_Loss import IIC_Loss
from DataLoaders.LoadPartASCAD import Datasetloader
from config import *
# 引入网络结构
# from Nets.net5g import ClusterNet5g
# from Nets.cnn_single-head import CNNNet
from Nets.Resnet import ResNet_18
from utils import setup_seed
from tqdm import tqdm

setup_seed(seed)
# show = ToPILImage()  # 可以把Tensor转成Image，方便可视化

def train():
    if (dataset_mode == 0):
        len_train = 50000
    elif (dataset_mode == 1):
        len_train = 10000
    # 无监督数据的加载
    train_loader, _ = Datasetloader(train_data_path)(bs, is_shuffle, dataset_mode, 0, len_train)
    # 有监督数据的加载
    supervised_size = len_train // supervised_rate
    print(f"有标签数据为前{supervised_size}个")
    labeled_train_loader, _ = Datasetloader(train_data_path)(bs + bs // supervised_rate, is_shuffle, dataset_mode, 0, supervised_size)
    labeled2_train_loader, _ = Datasetloader(train_data_path)(bs // supervised_rate, is_shuffle, dataset_mode, 0, supervised_size)

    print(f"无监督数据加载长度:{len(train_loader)} 起点训练加载长度:{len(labeled_train_loader)} 有监督数据加载长度:{len(labeled2_train_loader)}")
    
    # 更改网络结构在这里
    # model = CNNNet().to(device)
    model = ResNet_18().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    iic_loss_fn = IIC_Loss()
    iic_loss_fn = iic_loss_fn.to(device)
    # 交叉熵损失函数
    cross_loss_fn = nn.CrossEntropyLoss()
    cross_loss_fn = cross_loss_fn.to(device)

    transformer = TripleBatchTransform()
    transformer = transformer.to(device)

    # 使用有监督设置模型无监督训练时的起点
    with open(sur_lossSavePath, 'w') as f:
        for t_epoch in range(labeled_epochs):
            model.train()
            running_loss = 0.0
            print(f"labeled_epoch: {t_epoch}")
            for i, data in enumerate(tqdm(labeled_train_loader), 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                input_trans = transformer(inputs)
                optimizer.zero_grad()
                outputs = model(inputs)
                outputs_trans = model(input_trans)
                # 对不同变换的数据都进行损失更新
                loss1 = cross_loss_fn(outputs, labels)
                loss2 = cross_loss_fn(outputs_trans, labels)
                loss = loss1 + loss2
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Loss: {running_loss / len(train_loader)}")
            f.write(str(running_loss / len(train_loader)) + "\n")
            # 保存权重
            if not os.path.exists(sur_modelsavename):
                os.makedirs(sur_modelsavename)
            torch.save(model.state_dict(), sur_modelsavename + "epoch_" + str(t_epoch) + ".pth")

    with open(lossSavePath, 'w') as f:
        for epoch in range(num_epochs):
            running_loss = 0.0
            print(f"epoch: {epoch}")
            for i, (data, ldata) in enumerate(tqdm(zip(train_loader, labeled2_train_loader)), 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                linputs, llabels = ldata
                linputs, llabels = linputs.to(device), llabels.to(device)
                # 连接有监督数据和无监督数据
                inputs = torch.cat((inputs, linputs), 0)
                labels = torch.cat((labels, llabels), 0)
                # 通过数据变换对象对数据进行变换
                linputs_trans = transformer(linputs)
                inputs_trans = transformer(inputs)
                optimizer.zero_grad()
                # 有监督训练
                sur_outputs1 = model(linputs)
                sur_outputs2 = model(linputs_trans)
                sur_loss1 = cross_loss_fn(sur_outputs1, llabels)
                sur_loss2 = cross_loss_fn(sur_outputs2, llabels)
                sur_loss = (sur_loss1 + sur_loss2)
                # 无监督训练
                outputs1 = model(inputs)
                outputs2 = model(inputs_trans)
                #计算IID损失
                loss = 0.0
                loss_no_lamb = 0.0
                loss1, loss1_no_lamb = iic_loss_fn(outputs1, outputs2)
                loss += unsur_loss_rate * loss1
                print(f"unsur_loss: {loss} sur_loss: {sur_loss}")
                loss += sur_loss
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            f.write(str(running_loss / len(train_loader)) + '\n')
            print(f"Loss: {running_loss / len(train_loader)}")
            if (saveModel == True):
                # 如果saveModelPath路径不存在，创建路径
                if not os.path.exists(saveModelPath):
                    os.makedirs(saveModelPath)
                torch.save(model, saveModelPath + f"model_{epoch}.pth")
    # 保存模型
    print(f"最终模型保存在{modelsaveName}")
    torch.save(model, modelsaveName)
    # 保存有监督loss
    with open(lossSavePath + "l", "w") as f:
        for i in labeled_loss_list:
            f.write(str(i) + "\n")
    
train()
