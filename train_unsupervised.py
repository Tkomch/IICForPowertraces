import sys
import os
import torch
import torch.nn as nn
# 可视化张量图片
# from torchvision.transforms import ToPILImage
# from DataTransers.ImageTranser import DoubleBatchTransform
from DataTransers.TraceTranser import TripleBatchTransform
from IIC_Loss import IIC_Loss
from DataLoaders.LoadASCAD import Datasetloader
from config import *
# 引入网络结构
# from Nets.net5g import ClusterNet5g
from Nets.cnn_multi-head import CNNNet
from utils import setup_seed
from tqdm import tqdm

setup_seed(seed)
# show = ToPILImage()  # 可以把Tensor转成Image，方便可视化

def train():
    train_loader, _ = Datasetloader(train_data_path)(bs, is_shuffle, dataset_mode)
    # 更改网络结构在这里
    model = CNNNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    iic_loss_fn = IIC_Loss()
    iic_loss_fn = iic_loss_fn.to(device)

    transformer2 = TripleBatchTransform()
    transformer2 = transformer2.to(device)
    # 将每个epoch的loss保存下来
    loss_list = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print(f"epoch: {epoch}")
        for i, data in enumerate(tqdm(train_loader), 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # 用于测试数据变换效果
            # show(inputs[0]).show()
            # inputs_test1, inputs_test2 = transformer2(inputs)
            # show(inputs_test1[0]).show()
            # show(inputs_test2[0]).show()
            # exit()
            # 通过数据变换对象对数据进行变换
            inputs_trans2, inputs_trans3 = transformer2(inputs)
            optimizer.zero_grad()
            outputs1 = model(inputs)
            outputs2 = model(inputs_trans2)
            outputs3 = model(inputs_trans3)
            #计算IID损失
            loss = 0.0
            loss_no_lamb = 0.0
            for i in range(num_sub_heads):
                loss1, loss1_no_lamb = iic_loss_fn(outputs1[i], outputs2[i])
                loss2, loss2_no_lamb = iic_loss_fn(outputs1[i], outputs3[i])
                loss += (loss1 + loss2)
                loss_no_lamb += (loss1_no_lamb + loss2_no_lamb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        loss_list.append(running_loss/len(train_loader))
        print(f"Loss: {running_loss / len(train_loader)}")
        if (saveModel == True):
            # 如果saveModelPath路径不存在，创建路径
            if not os.path.exists(saveModelPath):
                os.makedirs(saveModelPath)
            torch.save(model, saveModelPath + f"model_{epoch}.pth")
    # 保存模型
    print(f"最终模型保存在{modelsaveName}")
    torch.save(model, modelsaveName)
    # 保存loss
    with open(lossSavePath, "w") as f:
        for i in loss_list:
            f.write(str(i) + "\n")
    
train()
