import sys
import os
import torch
import torch.nn as nn
# 可视化张量图片
from DataTransers.TraceTranser import TripleBatchTransform
from IIC_Loss import IIC_Loss
from DataLoaders.LoadPartASCAD import Datasetloader
from config import *
# 引入网络结构
from utils import setup_seed
from tqdm import tqdm

setup_seed(seed)
# show = ToPILImage()  # 可以把Tensor转成Image，方便可视化

def train():
    train_loader, _ = Datasetloader(train_data_path)(bs, is_shuffle, dataset_mode, left, right)
    # 更改网络结构在这里
    print("将使用%s网络结构进行训练" % net_structure)
    if (net_structure == 'cm3'):
        from Nets.cnn_multi_head import CNNNet
        model = CNNNet().to(device)
    if (net_structure == 'mm5'):
        from Nets.mlp_mutil_head_5layer import MLPNet
        model = MLPNet().to(device)
    if (net_structure == 'mm7'):
        from Nets.mlp_mutil_head_7layer import MLPNet
        model = MLPNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    iic_loss_fn = IIC_Loss()
    iic_loss_fn = iic_loss_fn.to(device)

    # transformer2 = TripleBatchTransform()
    # transformer2 = transformer2.to(device)
    # 将每个epoch的loss保存下来
    loss_list = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print(f"epoch: {epoch}")
        for i, data in enumerate(tqdm(train_loader), 0):
            inputs, inputs_trans, labels = data
            inputs, inputs_trans, labels = inputs.to(device), inputs_trans.to(device), labels.to(device)
            # 通过数据变换对象对数据进行变换
            # inputs_trans = transformer2(inputs)
            optimizer.zero_grad()
            outputs1 = model(inputs)
            outputs2 = model(inputs_trans)
            #计算IID损失
            loss = 0.0
            loss_no_lamb = 0.0
            for i in range(num_sub_heads):
                loss1, _ = iic_loss_fn(outputs1[i], outputs2[i])
                loss += loss1
                # loss_no_lamb += (loss1_no_lamb + loss2_no_lamb)
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
