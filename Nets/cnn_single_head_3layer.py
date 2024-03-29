import torch.nn as nn
import sys
sys.path.append('..')
from config import *

## the pre-trained model
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.front_fc = nn.Linear(1, 1)
        # the encoder part
        self.features = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=1, stride=1),
            nn.SELU(),
            nn.BatchNorm1d(8),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Conv1d(8, 16, kernel_size=11, stride=1),
            nn.SELU(),
            nn.BatchNorm1d(16),
            nn.AvgPool1d(kernel_size=11, stride=11),
            nn.Conv1d(16, 32, kernel_size=2, stride=1),
            nn.SELU(),
            nn.BatchNorm1d(32),
<<<<<<< Updated upstream
            nn.AvgPool1d(kernel_size=3, stride=3),
=======
>>>>>>> Stashed changes
            nn.Flatten()
        )

        # # the fully-connected layer 1
        # self.classifier_1 = nn.Sequential(
        #     nn.Linear(288, 2), # need change 700->288
        #     nn.SELU(),
        # )
        # # the output layer
        # self.final_classifier = nn.Sequential(
        #     nn.Linear(2, output_k)
        # )

        self.head = nn.Sequential(
            nn.Linear(288, 20), # need change 700->288
            nn.Linear(20, output_k),
            nn.Softmax(dim=1)
        )
    
    # how the network runs
    def forward(self, input):
        input = input.view(input.size(0), 1, input.size(1))
        x = self.features(input)
        x = self.head(x)
        return x
