import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        res = x
        ret = []
        for layer in model:
            res = layer(res)
            if isinstance(layer,torch.nn.modules.ReLU):
                frac = torch.sum(torch.all(res==0,axis=0))/res.size(1)
                ret.append(frac)
        return ret
        
    def suggest_fix(self, dead_fractions: List[float]) -> str:
        if torch.any(torch.tensor(dead_fractions)>0.5): return 'use_leaky_relu'
        if dead_fractions[0] > 0.3: return 'reinitialize'
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        flag = True
        for i in range(0,len(dead_fractions)-1):
                if dead_fractions[i] >= dead_fractions[i+1]: flag = False
        if dead_fractions[-1] > 0.1 and flag: return 'reduce_learning_rate'
        if max(dead_fractions) < 0.1: return 'healthy'
        return 'healthy'
