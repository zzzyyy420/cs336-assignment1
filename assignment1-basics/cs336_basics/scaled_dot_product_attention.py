import torch
import torch.nn as nn
from cs336_basics.softmax import softmax

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
        d_k = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * torch.rsqrt(torch.tensor(d_k))
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf')) #如果mask为0，则将对应位置的score设置为负无穷
        attn_weights = softmax(scores, dim=-1) #对key这一维度进行softmax归一化
        return attn_weights @ V # 将attn_weights与value相乘得到最终的输出