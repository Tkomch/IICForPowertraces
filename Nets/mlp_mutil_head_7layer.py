import torch.nn as nn
import sys
sys.path.append('..')
from config import *

## the pre-trained model
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(pca_dim, 64), # need change 700->288
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, pca_dim),
            nn.ReLU(),
        )

        self.head = nn.ModuleList([nn.Sequential(
            nn.Linear(pca_dim, pca_dim),
            nn.ReLU(),
            nn.Linear(pca_dim, output_k),
            nn.Softmax(dim=1)
            ) for _ in range(num_sub_heads)])
    
    # how the network runs
    def forward(self, input):
        result = []
        x = self.feature(input)
        for i in range(num_sub_heads):
            result.append(self.head[i](x))
        return result
