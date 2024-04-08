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
pca_dim = 96
# 数据集加载时将重复几次(默认为1)
dataset_count = 1
# 优化器 'SGD', 'adam'
opt_structure = 'adam'
# 模型结构 有监督：'cs3' 'cs4' 'resnet18' 'ms5' 无监督：'mm5' 'mm7' 'cmp3'
net_structure = 'cmp3'
modelsaveName = "Models/unsupervised_cmp3_03/model_2999.pth"
# sur_modelsavename = "Models/supervised/mm5_e30.pth"
# 是否在每个epoch保存模型
saveModel = False
# 是否是以权重的形式保存模型
save_weight = False
# 如果每个epoch保存模型，保存模型的路径
saveModelPath = "Models/unsupervised_cmp3_03/"
# Rank图的保存位置
rank_save_path = "Rank/rank_unspervised_cmp3_03_m2999_h4_p1.png"
# 测试时选取的头
chosen_head = 4
# 将结果交换的顺序
# output_order = [0, 1, 2, 3, 4, 5, 6, 7, 8]
# output_order = [2, 6, 0, 3, 4, 5, 1, 7, 8]  # cmp3_02_m0_h1
# output_order = [5, 1, 2, 8, 4, 0, 7, 6, 3]  # cmp3_02_m0_h4
# output_order = [2, 3, 0, 1, 4, 8, 5, 6, 7]  # cmp3_03_m2822_h4 cmp3_03_m277_h4
output_order = [2, 3, 0, 4, 1, 8, 5, 6, 7]  # cmp3_03_m277_h4
# output_order = [2, 3, 5, 4, 0, 1, 8, 6, 7]  # cmp3_03_m2999_h4
# output_order = [2, 3, 0, 1, 4, 8, 5, 6, 7]  # cmp3_03_m200_h4
# Loss保存路径
lossSavePath = "Metrics/Loss_unsupervised_cmp3_03.txt"
sur_lossSavePath = "Metrics/Loss_supervised_mm5_e30.txt"
# acc保存路径
accSavePath = "Metrics/Acc-Epochs_unsupervised_cmp3_03.txt"
# 计算设备
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = "cpu"
# 训练参数
num_epochs = 3000
# 起点训练epoch
# labeled_epochs = 2700
learning_rate = 0.0001
bs = 1
is_shuffle = False
# 有监督数据比例
# supervised_rate = 2
# 互信息最大损失占比
# unsur_loss_rate = 2
# 模型输出阈值
threshold = 0.2
# 随机数种子
seed = 42
# 模型分类头数量
num_sub_heads = 5
# 类别数量
output_k = 9
# 标注类别数
label_type = 9
# 变换中pca维度
transpca_dim = 96
# 固定平移
fixed_move = 48
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
