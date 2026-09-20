import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.ip = nn.Linear(784,512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(512,10)
        self.sigmoid = nn.Sigmoid()
        self.layers = [
            self.ip,
            self.relu,
            self.dropout,
            self.linear,
            self.sigmoid
        ]

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        res = images
        for layer in self.layers: res = layer(res)
        return res