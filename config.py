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
modelsaveName = "Models/semi-supervised_resnet-18_03"
sur_modelsavename = "Models/supervised/resnet-18_03/weight_"
# 是否在每个epoch保存模型
saveModel = True
# 是否是以权重的形式保存模型
save_weight = True
# 如果每个epoch保存模型，保存模型的路径
saveModelPath = "Models/semi-supervised_resnet-18_03/"
# Loss保存路径
lossSavePath = "Metrics/Loss_semi-supervised_resnet-18_03.txt"
sur_lossSavePath = "Metrics/Loss_supervised_resnet-18_03.txt"
# acc保存路径
accSavePath = "Metrics/Acc-Epochs_semi-supervised_resnet-18_01.txt"
# 计算设备
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = "cpu"
# 训练参数
num_epochs = 4000
# 起点训练epoch
labeled_epochs = 2700
learning_rate = 0.0001
bs = 32
is_shuffle = True
# 有监督数据比例
supervised_rate = 2
# 互信息最大损失占比
unsur_loss_rate = 2
# 模型输出阈值
threshold = 0.8
# 对图片来说输入图片大小
resize_shape_x = 64
resize_shape_y = 64
# 随机数种子
seed = 42
# 批归一化是否跟踪(IIC原网络参数)
batchnorm_track = True
# 对图片来说输入通道大小
in_channels = 3
# 模型分类头数量
num_sub_heads = 1
# 类别数量
output_k = 256
# 标注类别数
label_type = 256
# 是否使用小波变换(小波变换代码无法复用)
# wave_flag = False
# get_level = 1
# max_level = 5
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
