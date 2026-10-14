import torch
from torch import nn

from cs336_basics.SwiGLU import SwiGLU

from .causal_multi_head_attention_no_weight import CausalMultiHeadAttentionNoWeight


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int, theta: float, device=None):
        super().__init__()
        self.swiglu = SwiGLU(d_model, d_ff).to(device=device)
        self.causal_multi_head_attention = CausalMultiHeadAttentionNoWeight(
            d_model, n_heads, max_seq_len, theta, device=device
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.shape[-2], device=x.device)
        x = x + self.causal_multi_head_attention(x, positions)
        return x + self.swiglu(x)
