import torch.nn as nn
import sys
sys.path.append('..')
from config import *

## the pre-trained model
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(pca_dim, 64), # need change 700->288
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_k),
            nn.Softmax(dim=1)
        )
    
    # how the network runs
    def forward(self, input):
        x = self.head(input)
        return x
