import torch
import einops
import torch.nn as nn
import math
import torch.nn.functional as F
from cs336_basics.softmax import softmax
from cs336_basics.rope import RoPE


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, theta, max_seq_len, device):
        super(MultiHeadSelfAttention, self).__init__()
        self.rope = RoPE(theta=theta, d_k=d_model//n_heads, max_seq_len=max_seq_len, device=device)
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

    def attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
        d_k = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * torch.rsqrt(torch.tensor(d_k))
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf')) #如果mask为0，则将对应位置的score设置为负无穷
        attn_weights = softmax(scores, dim=-1) #对key这一维度进行softmax归一化
        return attn_weights @ V # 将attn_weights与value相乘得到最终的输出
    def forward(self, x, wq, wk, wv, wo, token_positions=None)->torch.Tensor:
        seq_len = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        else:
            token_positions = token_positions.to(device=x.device)
            
        token_positions = token_positions.unsqueeze(-2)

        q = x @ wq.T # (batch_size, seq_len, d_model) @ (d_model, d_k) -> (batch_size, seq_len, d_k)
        k = x @ wk.T # (batch_size, seq_len, d_model) @ (d_model, d_k) -> (batch_size, seq_len, d_k)
        v = x @ wv.T # (batch_size, seq_len, d_model) @ (d_model, d_v) -> (batch_size, seq_len, d_v)

        q = einops.rearrange(q, '... s (h d) -> ... h s d', h=self.n_heads)
        k = einops.rearrange(k, '... s (h d) -> ... h s d', h=self.n_heads)
        v = einops.rearrange(v, '... s (h d) -> ... h s d', h=self.n_heads)

        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)


        #现在的形状是(batch_size, n_heads, seq_len, head_dim)
        # 创建mask，用于防止当前位置的token看到未来的token。
        mask = torch.triu(torch.ones(seq_len, seq_len,dtype=torch.bool,device=x.device), diagonal=1)
        mask = mask.T
        mask.fill_diagonal_(True)

        out = self.attention(q, k, v, mask)
        out = einops.rearrange(out, '... h s d -> ... s (h d)')
        out = out @ wo.T
        return out
