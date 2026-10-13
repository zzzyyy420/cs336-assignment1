import torch
import torch.nn as nn
import einops

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even")
        self.theta = theta #这个是RoPE的底数超参数，不是直接的角度
        self.d_k = d_k 
        self.max_seq_len = max_seq_len
        self.device = device
        #计算频率
        freqs = 1.0 / (self.theta ** (torch.arange(0,self.d_k//2,device=device).float()*2 / self.d_k))         
        #记录每个token的位置信息
        positions = torch.arange(0, self.max_seq_len, device=device)
        #计算正弦和余弦
        sinusoids = positions.unsqueeze(-1)* freqs.unsqueeze(0) 
        self.register_buffer("cos_cache", sinusoids.cos(), persistent=False)
        self.register_buffer("sin_cache", sinusoids.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:        # 这里的x是输入的稠密向量，token_positions是token的位置信息
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        x_part1 = x[..., 0::2]
        x_part2 = x[..., 1::2]

        output1 = x_part1 * cos - x_part2 * sin # 偶数位置乘以cos，奇数位置乘以sin
        output2 = x_part1 * sin + x_part2 * cos # 偶数位置乘以sin，奇数位置乘以cos
        
        out = torch.stack([output1, output2], dim=-1)  
        out = einops.rearrange(out, '... d1 d2 -> ... (d1 d2)')  # 将最后两维合并
        return out
