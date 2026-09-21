import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2/(fan_in+fan_out))
        return (torch.randn(fan_out,fan_in)*std).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2/(fan_in))
        return (torch.randn(fan_out,fan_in)*std).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        torch.manual_seed(0)
        layers = []
        fan_in,fan_out = input_dim,hidden_dim
        ans = []
        for idx in range(num_layers):
            if init_type == 'xavier':
                std = math.sqrt(2/(fan_in+fan_out))
                weight_mat = torch.randn(fan_out,fan_in)*std
            elif init_type == 'kaiming':
                std = math.sqrt(2/(fan_in))
                weight_mat = torch.randn(fan_out,fan_in)*std
            else:
                weight_mat = torch.randn(fan_out,fan_in)
            layers.append(weight_mat.clone())
            fan_in,fan_out = fan_out,fan_out
        inp = torch.randn(input_dim)
        res = inp
        for layer in layers:
            res = layer @ res
            res = torch.where(res > 0, res, torch.tensor(0.0))
            ans.append(round(torch.std(res).item(),2))
        return ans
