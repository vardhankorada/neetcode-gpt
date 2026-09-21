import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        ans = []
        with torch.no_grad():
            res = x
            for layer in model:
                res = layer(res)
                if isinstance(layer,torch.nn.modules.linear.Linear):
                    mean = torch.mean(res)
                    std = torch.std(res)
                    dead_frac = torch.sum(torch.all(res<=0,axis=0))/res.size(1)
                    ans.append({"mean":mean.item(),"std":std.item(),"dead_fraction":dead_frac.item()})
        return ans

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        ans = []
        model.zero_grad()
        y_pred = model(x)
        loss_met = nn.MSELoss()
        loss = loss_met(y_pred,y)
        loss.backward()
        for layer in model:
            if isinstance(layer,torch.nn.modules.linear.Linear):
                grad = layer.weight.grad
                mean = torch.mean(grad)
                std = torch.std(grad)
                norm = torch.norm(grad)
                ans.append({
                    "mean":mean.item(),
                    "std":std.item(),
                    "norm":norm.item()
                })
        return ans

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        for layer in activation_stats:
            if layer['dead_fraction'] > 0.5: return 'dead_neurons'
        for dic in gradient_stats:
            if dic['norm'] > 1000: return 'exploding_gradients'
        if gradient_stats[-1]['norm'] < 1e-5: return 'vanishing_gradients'
        for dic in activation_stats:
            if dic['std'] < 0.1 : return 'vanishing_gradients'
            if dic['std'] > 10.0 : return 'exploding_gradients'
        return 'healthy'
