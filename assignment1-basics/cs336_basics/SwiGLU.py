import torch
from torch import nn
from cs336_basics.linear import LinearModule
class SwiGLU(nn.Module):
    """
    SwiGLU 是激活函数，它通过将输入乘以sigmoid函数，然后乘以一个线性变换来得到输出。
    公式是：
    out = w2(w1(x) * sigmoid(w1(x)) * w3(x))
    其中x是输入，w1(x)是线性变换，sigmoid(w1(x))是sigmoid函数，w2(x)是线性变换，w3(x)是线性变换。
    Args:
        d_model (int): 输入的维度
        d_ff (int): 隐藏维度
    input:
        x: (batch_size, seq_len, d_model) 
    output:
        out: (batch_size, seq_len, d_model) 
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = LinearModule(d_model, d_ff)
        self.w2 = LinearModule(d_ff, d_model)
        self.w3 = LinearModule(d_model, d_ff)


    def silu(self, x):
        return x * torch.sigmoid(x)
    
    def forward(self, x):
        return self.w2(self.silu(self.w1(x)) * self.w3(x))