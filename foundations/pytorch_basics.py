import torch
import torch.nn
from torchtyping import TensorType

# Round all answers to 4 decimal places: torch.round(tensor, decimals=4)
class Solution:
    def reshape(self, x: TensorType[float]) -> TensorType[float]:
        # Reshape (M, N) tensor to (M*N/2, 2)
        # Use torch.reshape(tensor, new_shape)
        return torch.reshape(x,(x.size(0)*x.size(1)//2,2))

    def average(self, x: TensorType[float]) -> TensorType[float]:
        # Compute column-wise mean (average across rows)
        # Use torch.mean(tensor, dim=0)
        return torch.mean(x,dim=0)

    def concatenate(self, x: TensorType[float], y: TensorType[float]) -> TensorType[float]:
        # Join two tensors side-by-side along dim=1
        # Use torch.cat((a, b), dim=1)
        return torch.cat((x,y),dim=1)

    def get_loss(self, preds: TensorType[float], tgt: TensorType[float]) -> TensorType[float]:
        # Compute Mean Squared Error between prediction and target
        # Use torch.nn.functional.mse_loss(prediction, target)
        return torch.nn.functional.mse_loss(preds,tgt)
