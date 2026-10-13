import torch
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True)
    x_exp = torch.exp(x - x_max.values)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)