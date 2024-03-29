import h5py
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
# import pywt
import sys
sys.path.append('..')
from config import *

class Datasetloader():
    def __init__(self, data_path):
        self.data_file_path = data_path

    # 小波变换
    # def wavelet_decompose(self, data, get_level, max_level=5):
    #     result = []
    #     coeffs = pywt.wavedec(data, 'db4', level=max_level)
    #     cA5, cD5, cD4, cD3, cD2, cD1 = coeffs
    #     # 小波重构
    #     result.append(pywt.waverec(np.multiply(coeffs, [1, 0, 0, 0, 0, 0]).tolist(), 'db4'))
    #     result.append(pywt.waverec(np.multiply(coeffs, [0, 1, 0, 0, 0, 0]).tolist(), 'db4'))
    #     result.append(pywt.waverec(np.multiply(coeffs, [0, 0, 1, 0, 0, 0]).tolist(), 'db4'))
    #     result.append(pywt.waverec(np.multiply(coeffs, [0, 0, 0, 1, 0, 0]).tolist(), 'db4'))
    #     result.append(pywt.waverec(np.multiply(coeffs, [0, 0, 0, 0, 1, 0]).tolist(), 'db4'))
    #     result.append(pywt.waverec(np.multiply(coeffs, [1, 0, 0, 0, 0, 1]).tolist(), 'db4'))
    #     return result[get_level]

    def __call__(self, bs, shuffle_, mode):
        in_file = h5py.File(self.data_file_path, 'r')
        plain_text_ = None
        if (mode == 0):
            traces_profiling = np.array(in_file['Profiling_traces/traces'])
            labels_profiling = np.array(in_file['Profiling_traces/labels'])
            metadatas = in_file['Profiling_traces/metadata']
            plain_text = []
            for md in metadatas:
                plain_text.append(md[0][2])
            plain_text_ = np.array(plain_text)
        else:
            traces_profiling = np.array(in_file['Attack_traces/traces'])
            labels_profiling = np.array(in_file['Attack_traces/labels'])
    
            metadatas = in_file['Attack_traces/metadata']
            plain_text = []
            for md in metadatas:
                plain_text.append(md[0][2])
            plain_text_ = np.array(plain_text)
    
        # 输出labels_profiling的形状
        print(labels_profiling.shape)
        plain_text_ = torch.from_numpy(plain_text_)
        # if wave_flag:
        #     traces_profiling = np.array([self.wavelet_decompose(traces_profiling[i], get_level, max_level) for i in range(len(traces_profiling))])
        #     print(traces_profiling.shape)
        traces, labels = torch.from_numpy(traces_profiling).float(), torch.from_numpy(labels_profiling)
        labels = self.convert_labels(labels, label_type)
        print(f"labels shape: {labels.shape}")
        print(labels)
        ascad_dataset = TensorDataset(traces, labels)
        loader = DataLoader(dataset=ascad_dataset, batch_size=bs, shuffle=shuffle_)
        return loader, plain_text_

    # 将labels值转为其他分类的labels
    def convert_labels(self, labels, label_type):
        for i, label in enumerate(labels):
            if (label_type == 2):
                labels[i] = label >> 7
            elif (label_type == 4):
                labels[i] = label >> 6
            elif (label_type == 8):
                labels[i] = label >> 5
            elif (label_type == 16):
                labels[i] = label >> 4
            elif (label_type == 32):
                labels[i] = label >> 3
            elif (label_type == 64):
                labels[i] = label >> 2
            elif (label_type == 128):
                labels[i] = label >> 1
        return labels

# TEST
# if __name__ == '__main__':
#     datasetloader = Datasetloader(train_data_path)
#     loader, plain_text = datasetloader(1, True, 0)
#     for i, (data, label) in enumerate(loader):
#         print(label)
#         break
#     print(255 >> 7)

