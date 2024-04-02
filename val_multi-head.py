from time import sleep
import os
import sys
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from DataLoaders.LoadPartASCAD import Datasetloader
from config import *

def val():
    test_loader, _ = Datasetloader(test_data_path)(bs, is_shuffle, dataset_mode, left, right)
    # 加载模型
    if (saveModel == False):
        model = torch.load(modelsaveName, map_location=device)
        model.eval()
        model = model.to(device)
        acc_num_sub_heads = []
        # 随着曲线增加的准确率
        for i in range(num_sub_heads):
            correct = 0
            total = 0
            acc_traces = []

            t_total = 0
            t_correct = 0
            with torch.no_grad():
                for data in tqdm(test_loader):
                    images, images2, labels = data
                    images, images2, labels = images.to(device), images2.to(device), labels.to(device)
                    outputs = model(images)
                    
                    # 对输出进行阈值筛选，计算准确率
                    threshold_true_index = []
                    for bs_i, bs_data in enumerate(outputs[i]):
                        if (bs_data[0] > threshold or bs_data[1] > threshold):
                            threshold_true_index.append(bs_i)

                    _, predicted = torch.max(outputs[i].data, 1)

                    for p_i, p_data in enumerate(predicted):
                        if (p_i in threshold_true_index):
                            t_total += 1
                            if (p_data == labels[p_i]):
                                t_correct += 1
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    acc_traces.append(correct / total)
            acc_num_sub_heads.append(acc_traces)
            accuracy = correct / total
            threshold_accuracy = t_correct / t_total
            print(f"正确个数:{correct}/{total} {i}头测试总准确率: {accuracy * 100:.2f}%")
            print(f"正确个数:{t_correct}/{t_total} {i}头测试阈值准确率: {threshold_accuracy * 100:.2f}%")
        # 绘制acc-traces曲线
        for i in range(num_sub_heads):
            plt.plot(acc_num_sub_heads[i], label=f"{i} head")
        plt.xlabel("Traces")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
        print()
    else:
        m = os.listdir(saveModelPath)
        model_file = ["model_" + str(i) + ".pth" for i in range(0, len(m))]
        max_accuracy = 0.0
        max_model = ""
        accuracy_all = []
        for file in model_file:
            model = torch.load(saveModelPath + file, map_location=device)
            model.eval()
            model = model.to(device)
            acc_epoch = []
            for i in range(num_sub_heads):
                correct = 0
                total = 0
                t_correct = 0
                t_total = 0
                with torch.no_grad():
                    for data in tqdm(test_loader):
                        images, _, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        # 对输出进行阈值筛选，计算准确率
                        threshold_true_index = []
                        for bs_i, bs_data in enumerate(outputs[i]):
                            if (bs_data[0] > threshold or bs_data[1] > threshold):
                                threshold_true_index.append(bs_i)

                        _, predicted = torch.max(outputs[i].data, 1)

                        for p_i, p_data in enumerate(predicted):
                            if (p_i in threshold_true_index):
                                t_total += 1
                                if (p_data == labels[p_i]):
                                    t_correct += 1

                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                accuracy = correct / total
                threshold_accuracy = t_correct / t_total
                acc_epoch.append(accuracy)
                if (accuracy > max_accuracy):
                    max_accuracy = accuracy
                    max_model = file
                print(f"{file} {i}头测试准确率: {accuracy * 100:.2f}%")
                print(f"{file} {i}头测试阈值准确率: {threshold_accuracy * 100:.2f}%")
            accuracy_all.append(acc_epoch)
            print()
        # 将每个epoch的准确率保存到文件
        with open(accSavePath, "w") as f:
            for acc_epochs in accuracy_all:
                for acc in acc_epochs:
                    f.write(str(acc) + " ")
                f.write("\n")
        print(f"最高准确率模型: {max_model} 准确率: {max_accuracy * 100:.2f}%")

val()
