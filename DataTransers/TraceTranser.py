from torch import nn
import torch
import numpy as np
import sys
from sklearn.decomposition import PCA
sys.path.append('..')
from config import *

# 循环平移
class LoopMove():
    def __init__(self, move):
        self.move = move

    def __call__(self, x):
        # x shape [bs len]
        x_np = x.cpu().numpy()
        x_np = np.roll(x_np, self.move, axis=1)
        x = torch.from_numpy(x_np).to(device)
        return x

class RandomLoopMove():
    def __init__(self, min_move, max_move):
        self.min_move = min_move
        self.max_move = max_move

    def __call__(self, x):
        move = np.random.randint(self.min_move, self.max_move)
        loop_move = LoopMove(move)
        return loop_move(x)

# 整合, 将n个数整合成一个数，可以是平均值，最大值，最小值
# 然后进行邻近插值
class Integrate():
    def __init__(self, step, mode):
        self.step = step
        self.mode = mode

    def __call__(self, x):
        x_np = x.cpu().numpy()
        for i, data in enumerate(x_np):
            data = data.reshape(-1, self.step)
            ori_len = len(data)
            if self.mode == 'mean':
                data = np.mean(data, axis=1)
            elif self.mode == 'max':
                data = np.max(data, axis=1)
            elif self.mode == 'min':
                data = np.min(data, axis=1)
            for j, d in enumerate(data):
                x_np[i][2*j] = data[j]
                x_np[i][2*j+1] = data[j]
        x = torch.from_numpy(x_np).to(device)
        return x

# 加入高斯噪声
class AddGaussianNoise():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        x_np = x.cpu().numpy()
        noise = (torch.randn(x[0].size()) * self.std + self.mean).numpy()
        for i, data in enumerate(x_np):
            x_np[i] = data + noise
        return torch.from_numpy(x_np).to(device)

# 水平翻转
# class HorizontalFlip():
#     def __init__(self):
#         pass
# 
#     def __call__(self, x):
#         x_np = x.cpu().numpy()
#         for i, data in enumerate(x_np):
#             x_np[i] = data[::-1]
#         return torch.from_numpy(x_np).to(x.device)

# 标准化
class Normalize():
    def __init__(self):
        pass

    def __call__(self, x):
        x_np = x.cpu().numpy()
        for i, data in enumerate(x_np):
            data = (data - np.mean(data)) / np.std(data)
            x_np[i] = data
        return torch.from_numpy(x_np).to(device)
    
# TODO 小波变换
# class WaveTransform():
#     def __init__(self, get_level, max_level):
#         self.get_level = get_level
#         self.max_level = max_level
# 
#     # 小波变换
#     def wavelet_decompose(self, data, get_level, max_level=5):
#         result = []
#         coeffs = pywt.wavedec(data, 'db4', level=max_level)
#         cA5, cD5, cD4, cD3, cD2, cD1 = coeffs
#         # 小波重构
#         print(f"coeffs: {coeffs}")
#         print(f"coeffs shape: {coeffs.shape}")
#         exit()
#         result.append(pywt.waverec(np.multiply(coeffs, [1, 0, 0, 0, 0, 0]).tolist(), 'db4'))
#         result.append(pywt.waverec(np.multiply(coeffs, [0, 1, 0, 0, 0, 0]).tolist(), 'db4'))
#         result.append(pywt.waverec(np.multiply(coeffs, [0, 0, 1, 0, 0, 0]).tolist(), 'db4'))
#         result.append(pywt.waverec(np.multiply(coeffs, [0, 0, 0, 1, 0, 0]).tolist(), 'db4'))
#         result.append(pywt.waverec(np.multiply(coeffs, [0, 0, 0, 0, 1, 0]).tolist(), 'db4'))
#         result.append(pywt.waverec(np.multiply(coeffs, [1, 0, 0, 0, 0, 1]).tolist(), 'db4'))
#         return result[get_level]
# 
#     def __call__(self, x):
#         x_np = x.cpu().numpy()
#         for i, data in enumerate(x_np):
#             x_np[i] = np.array(self.wavelet_decompose(data, self.get_level, self.max_level))
#         return torch.from_numpy(x_np).to(x.device)
        

class TripleBatchTransform(nn.Module):
    def __init__(self):
        super(TripleBatchTransform, self).__init__()
        self.loop_move = RandomLoopMove(min_shift, max_shift)
        self.integrate = Integrate(i_step, i_mode)
        self.add_gaussian_noise = AddGaussianNoise(mean, std)
        # self.horizontal_flip = HorizontalFlip()
        self.normalize = Normalize()
        
    def __call__(self, x):
        x = self.loop_move(x)
        x = self.add_gaussian_noise(x)
        x = self.integrate(x)
        x = self.normalize(x)
        return x

class PCATransform(nn.Module):
    def __init__(self, pca_dim):
        self.PCA = PCA(pca_dim)

    def __call__(self, x):
        self.PCA.fit(x)
        x = self.PCA.transform(x)
        return x

class PCATransform2(nn.Module):
    def __init__(self, pca_dim):
        self.PCA = PCA(pca_dim)

    def __call__(self, x):
        self.PCA.fit(x)
        x = self.PCA.transform(x)
        # 将x的维度保持和pca_dim一致
        return x[:, -1 * pca_dim:]

class TripleBatchTransform2(nn.Module):
    def __init__(self):
        super(TripleBatchTransform2, self).__init__()
        self.loop_move = RandomLoopMove(min_shift, max_shift)
        self.integrate = Integrate(i_step, i_mode)
        self.add_gaussian_noise = AddGaussianNoise(mean, std)
        # self.horizontal_flip = HorizontalFlip()
        self.normalize = Normalize()
        
    def __call__(self, x):
        x1 = self.loop_move(x)
        x2 = self.add_gaussian_noise(x)
        x3 = self.normalize(x)
        x4 = self.integrate(x)
        # x5 = self.horizontal_flip(x)
        return x1, x2, x3, x4

# TEST
# if __name__ == "__main__":
#     # 生成8*700的tensor
#     x = torch.randn(8, 700)
#     triple_batch_transform = TripleBatchTransform()
#     x1, x2 = triple_batch_transform(x)
#     print(x)
#     print(f"x shape: {x.shape}")
#     print(x1)
#     print(f"x1 shape: {x1.shape}")
#     print(x2)
#     print(f"x2 shape: {x2.shape}")
