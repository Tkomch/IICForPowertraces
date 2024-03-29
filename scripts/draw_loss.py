import matplotlib.pyplot as plt
import numpy as np

log_flag = False
limmit_flag = False
limmit = -20000

# log路径
log_path = '/home/ning/Workspace/SchoolJob/scripts/code/基于不变信息分类/Metrics/Loss_supervised_256classes_CNN3layer_01.txt'
# log_path = 'experimental_data/YN/20240227-2自监督CNN-6trans-1000epochs分类.log'

def cal_normal_loss():
    if log_flag:
        # 读取开头为Loss的行
        with open(log_path, 'r') as f:
            lines = f.readlines()
            loss_lines = [line for line in lines if line.startswith('Loss')]
            # print(loss_lines)
            loss = []
            for line in loss_lines:
                loss_s = line.split(' ')[1]
                line_loss = float(loss_s)
                if limmit_flag:
                    if (line_loss < limmit):
                        line_loss = limmit
                if (loss_s[-1] != 'nan'):
                    loss.append(line_loss)
            # print(loss)
            # 绘图
            plt.plot(np.arange(len(loss)), loss)
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.title('Loss')
            plt.show()
    else:
        # 读取每行的loss绘制图像
        with open(log_path, 'r') as f:
            lines = f.readlines()
            loss = []
            for line in lines:
                line_loss = float(line)
                if limmit_flag:
                    if (line_loss < limmit):
                        line_loss = limmit
                loss.append(line_loss)
            plt.plot(np.arange(len(loss)), loss)
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.title('Loss')
            plt.show()

def cal_trans_loss():
    with open(log_path, 'r') as f:
        # 每行一个Loss的曲线 空格分割
        lines = f.readlines()
        loss = []
        for line in lines:
            line_loss = line.strip().split(' ')
            line_loss = [float(x) for x in line_loss]
            loss.append(line_loss)
        for i, l in enumerate(loss):
            plt.plot(l, label='transformation' + str(i))
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Loss')
        plt.legend()
        plt.show()

# cal_trans_loss()
cal_normal_loss()
