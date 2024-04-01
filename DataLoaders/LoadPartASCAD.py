import h5py
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
from sklearn.decomposition import PCA
import sys
sys.path.append('..')
from config import *
from DataTransers.TraceTranser import PCATransform
from DataTransers.TraceTranser import PCATransform2

class Datasetloader():
    def __init__(self, data_path):
        self.data_file_path = data_path

    def __call__(self, bs, shuffle_, mode, left, right):
        in_file = h5py.File(self.data_file_path, 'r')
        plain_text_ = None
        if (mode == 0):
            traces_profiling = np.array(in_file['Profiling_traces/traces'][left:right])
            labels_profiling = np.array(in_file['Profiling_traces/labels'][left:right])
            metadatas = in_file['Profiling_traces/metadata'][left:right]
            plain_text = []
            for md in metadatas:
                plain_text.append(md[0][2])
            plain_text_ = np.array(plain_text)
        else:
            traces_profiling = np.array(in_file['Attack_traces/traces'])[left:right]
            labels_profiling = np.array(in_file['Attack_traces/labels'])[left:right]
    
            metadatas = in_file['Attack_traces/metadata'][left:right]
            plain_text = []
            for md in metadatas:
                plain_text.append(md[0][2])
            plain_text_ = np.array(plain_text)
    
        # 输出labels_profiling的形状
        print(labels_profiling.shape)
        labels = torch.from_numpy(labels_profiling)
        plain_text_ = torch.from_numpy(plain_text_)
        labels = self.convert_labels(labels, label_type)
        print(f"labels shape: {labels.shape}")
        print(labels)

        if use_pca:
            print("使用PCA降维")
            pca_trans1 = PCATransform(pca_dim)
            pca_trans2 = PCATransform2(transpca_dim)
            traces_profiling1 = pca_trans1(traces_profiling)
            traces_profiling2 = pca_trans2(traces_profiling)
            traces, traces2 = torch.from_numpy(traces_profiling1).float(), torch.from_numpy(traces_profiling2).float()
            ascad_dataset = TensorDataset(traces, traces2, labels)
            loader = DataLoader(dataset=ascad_dataset, batch_size=bs, shuffle=shuffle_)
        else:
            traces = torch.from_numpy(traces_profiling).float()
            ascad_dataset = TensorDataset(traces, labels)
            loader = DataLoader(dataset=ascad_dataset, batch_size=bs, shuffle=shuffle_)
        return loader, plain_text_

    # 计算一个值的汉明重量
    def hamming_weight(self, num):
        count = 0
        while num:
            count += 1
            num &= num - 1
        return count

    # 将labels值转为其他分类的labels
    def convert_labels(self, labels, label_type):
        for i, label in enumerate(labels):
            if (label_type == 2):
                # 如果汉明重量等于4就废弃
                if (self.hamming_weight(label.item()) < 4):
                    labels[i] = 0
                elif (self.hamming_weight(label.item()) > 4):
                    labels[i] = 1
                else:
                    labels[i] = 0
            elif (label_type == 9):
                labels[i] = self.hamming_weight(label.item())
        return labels
