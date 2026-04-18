import torch
import torch.nn as nn
import torch.nn.functional as F

class QMoLE_Expert(nn.Module):
    """Ultra-Compressed Expert with a 16-dim bottleneck"""
    def __init__(self, hidden_size, intermediate_size=16):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.up_proj(F.silu(self.down_proj(x)))

class QMoLE_Layer(nn.Module):
    def __init__(self, hidden_size=64, num_experts=3, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([QMoLE_Expert(hidden_size) for _ in range(num_experts)])

    def forward(self, x):
        logits = self.router(x)
        weights = F.softmax(logits, dim=-1)
        top_weights, top_indices = torch.topk(weights, self.top_k, dim=-1)
        combined_output = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = top_indices[..., i]
            expert_weight = top_weights[..., i].unsqueeze(-1)
            for j in range(self.num_experts):
                mask = (expert_idx == j)
                if mask.any():
                    combined_output[mask] += self.experts[j](x)[mask] * expert_weight[mask]
        return combined_output