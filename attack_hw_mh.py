import torch
# import torch.nn.functional as F
import math
from DataLoaders.LoadPartASCAD import Datasetloader
from DataTransers.TraceTranser import TripleBatchTransform
from config import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from Nets.Resnet import ResNet_18

key_score = dict()
""" 密钥—分数字典 """

mval_key = dict()
""" 中间值-密钥字典 """

def hamming_weight(num):
    count = 0
    while num:
        count += 1
        num &= num - 1
    return count

def cal_middleval(p_text, key):
    """ 计算sbox(p[3] xor k[3])
    注意要传两个数字型
    """
    s_box = [[0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76],
             [0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0],
             [0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15],
             [0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75],
             [0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84],
             [0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF],
             [0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8],
             [0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2],
             [0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73],
             [0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB],
             [0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79],
             [0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08],
             [0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A],
             [0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E],
             [0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF],
             [0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16]]

    m_val = p_text ^ key
    row = m_val >> 4
    col = m_val & 0x0F
    mid_result = hamming_weight(int(s_box[row][col]))
    # 二分类
    if (label_type == 2):
        if (mid_result <= 4):
            return 0
        else:
            return 1
    elif (label_type == 9):
        return mid_result


if __name__ == "__main__":
    """ 初始化密钥-分数字典 """
    for i in range(256):
        key_score[i] = 0.0

    trans_func = TripleBatchTransform()
    test_data_loader, plain_texts = Datasetloader(test_data_path)(bs, is_shuffle, dataset_mode, left, right)

    if (save_weight == True):
        model = ResNet_18()
        model.load_state_dict(torch.load(modelsaveName, map_location=device))
    else:
        model = torch.load(modelsaveName, map_location=device)
    model.eval()
    model = model.to(device)

    # 计算准确度 #
    right_count = 0.0
    data_count = 0
    best_acc = 0.0

    first_report_flag = False

    rank_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_data_loader)):
            att_trace, att_trace2, att_label = data
            att_trace = att_trace.to(device)
            att_trace2 = att_trace2.to(device)
            att_label = att_label.to(device)
            # att_trace = trans_func(att_trace)
            outputs = model(att_trace)
            # 中间值分数 #
            # softmax_outputs = F.softmax(outputs, dim=1)
            softmax_outputs = outputs[chosen_head]

            # 生成中间值-密钥字典 #
            # 二分类
            for o in range(output_k):
                mval_key[o] = []
            for k in range(256):
                mval_key[cal_middleval(plain_texts[i].item(), k)].append(k)

            # 将中间值分数赋给密钥 #
            for mv in range(label_type):
                for softmax_output in softmax_outputs:
                    # 按照output_order的顺序换掉softmax_output的顺序
                    output_temp = softmax_output.clone()
                    for j in range(len(output_order)):
                        softmax_output[j] = output_temp[output_order[j]]
                    for k in mval_key[mv]:
                        a = key_score[k]
                        b = softmax_output[mv].item()
                        if (b != 0):
                            key_score[k] = a + math.log(b, 2)
                        else:
                            key_score[k] = a - 9999999

            # 对字典进行排序 #
            key_score_ = sorted(key_score.items(), key=lambda d: d[1], reverse=True)
            # 查找224的排名
            for j in range(256):
                if key_score_[j][0] == 224:
                    rank = j
                    break
            if (rank == 0 and first_report_flag == False):
                print(f"rank=0 : 曲线批次:{i}")
                first_report_flag = True
            rank_list.append(rank)
            # print(key_score_)
            # print(f"第{i + 1}条能量迹")
            # command = input()
            # if command == "q":
            #     break
            # elif command == "d":
            #     # 绘制rank-traces图
            #     plt.plot(rank_list)
            #     plt.show()

    plt.plot(rank_list)
    # plt.show()
    plt.savefig(rank_save_path)

            # predicted = torch.argmax(outputs, dim=1).item()

            # # Test #
            # if (i == 2):
            #     print(plain_texts[i])
            #     print(cal_middleval(plain_texts[i].item(), 224))
            #     print(att_label[0].item())
            #     print(predicted)
            #     exit()

            # if (predicted == att_label.item()):
            #     right_count += 1
            # print('\r', f"进度：{i + 1}/{len(test_data_loader)}", end="")
            # data_count += 1
    # accuracy = right_count / data_count
    # print(f" 准确度: {accuracy}")
