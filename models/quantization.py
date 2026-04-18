import torch
import torch.nn as nn
import torch.nn.functional as F

class BitLinear(nn.Linear):
    """1.58-bit (ternary) linear layer logic as implemented in QMoLE"""
    def forward(self, x):
        w = self.weight
        scale = w.abs().mean()
        # Ternary weight quantization to {-1, 0, 1}
        w_quant = torch.round(w / (scale + 1e-7)).clamp(-1, 1)
        # Straight-Through Estimator (STE)
        w_quant = w + (w_quant - w).detach()
        x_norm = F.layer_norm(x, (x.shape[-1],))
        return F.linear(x_norm, w_quant, self.bias)