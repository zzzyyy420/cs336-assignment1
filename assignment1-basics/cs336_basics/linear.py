import torch
from torch import nn

"""在讲义中，线性层在最后一步将d_model维度映射到vocab_size维度"""
class LinearModule(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.W = nn.Parameter(torch.empty(self.out_features, self.in_features, device=self.device, dtype=self.dtype))
        # self.b = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))

        # 对权重进行Xavier初始化
        std = (2 / (self.in_features + self.out_features) )** 0.5
        torch.nn.init.trunc_normal_(self.W, std=std, a = -3 * std, b = 3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T