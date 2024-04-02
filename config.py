import torch

# 定义路径
# data_root = "/home/ning/Workspace/SchoolJob/scripts/code/datasets/猫狗数据集/"
# train_data_path = (data_root + 'training_set/training_set/')
# test_data_path = (data_root + 'test_set/test_set/')
data_root = "/home/ning/workspace/datasets/sidechannel/"
train_data_path = (data_root + 'ASCAD.h5')
test_data_path = (data_root + 'ASCAD.h5')
# 对ASCAD数据集来说加载训练集还是测试集
dataset_mode = 0
# 加载数据集的范围
left = 0
right = 50000
# 是否使用PCA变换
use_pca = True
# 若使用PCA变换，PCA的维度
pca_dim = 24
# 数据集加载时将重复几次(默认为1)
dataset_count = 1
# 模型结构 有监督：'cs3' 'cs4' 'resnet18' 'ms5' 无监督：'mm5' 'mm7'
net_structure = 'mm5'
modelsaveName = "/home/ning/workspace/基于不变信息分类/Models/unsupervised_mm5_04.pth"
# sur_modelsavename = "Models/supervised/resnet-18_03/weight_"
# 是否在每个epoch保存模型
saveModel = True
# 是否是以权重的形式保存模型
save_weight = False
# 如果每个epoch保存模型，保存模型的路径
saveModelPath = "Models/unsupervised_mm5_04/"
# Rank图的保存位置
rank_save_path = "Rank/rank_unspervised_mm7_01_h0_p1.png"
# 测试时选取的头
chosen_head = 0
# 将结果交换的顺序
output_order = [1, 0]
# Loss保存路径
lossSavePath = "Metrics/Loss_unsupervised_mm5_04.txt"
# sur_lossSavePath = "Metrics/Loss_supervised_resnet-18_03.txt"
# acc保存路径
accSavePath = "Metrics/Acc-Epochs_unsupervised_mm5_04.txt"
# 计算设备
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = "cpu"
# 训练参数
num_epochs = 1500
# 起点训练epoch
# labeled_epochs = 2700
learning_rate = 0.00001
bs = 32
is_shuffle = True
# 有监督数据比例
# supervised_rate = 2
# 互信息最大损失占比
# unsur_loss_rate = 2
# 模型输出阈值
threshold = 0.8
# 随机数种子
seed = 73
# 模型分类头数量
num_sub_heads = 5
# 类别数量
output_k = 2
# 标注类别数
label_type = 2
# 变换中pca维度
transpca_dim = 48
# 循环平移的最大与最小值
min_shift = 1
max_shift = 10
# 整合步长(尽量不要更改，更改的话需要更改50和51行代码)
i_step = 2
# 整合方式
i_mode = 'mean'
# 高斯噪声参数
mean = 0
std = 0.1
