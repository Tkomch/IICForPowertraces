"""
从文件中读取acc数据，绘制acc-epochs曲线
"""

import matplotlib.pyplot as plt
import sys

acc_file_path = "/home/ning/Workspace/SchoolJob/scripts/code/基于不变信息分类/Metrics/Acc-Epochs_semi-supervised_resnet-18_01.txt"

def cal_normal_acc():
    # 读取文件，每行num_sub_heads个数据
    with open(acc_file_path, "r") as f:
        lines = f.readlines()
        accs = []
        for line in lines:
            accs.append([float(x) for x in line.split()])
        print(accs)
        # 将每组数据绘制成折线图
        for i in range(len(accs[0])):
            acc = [x[i] for x in accs]
            plt.plot(acc, label=f"{i} head")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()

cal_normal_acc()
