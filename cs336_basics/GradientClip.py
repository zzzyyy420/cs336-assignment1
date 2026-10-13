import torch


class GradientClip:
    def __init__(self, parameters, max_l2_norm, epslion=1e-6):
        self.parameters = list(parameters)
        self.max_l2_norm = max_l2_norm
        self.epslion = epslion

    @torch.no_grad()
    def __call__(self):
        grads = [p.grad for p in self.parameters if p.grad is not None]
        if not grads:
            return

        norm_dtype = torch.float64 if grads[0].dtype == torch.float64 else torch.float32
        total_squared_norm = torch.zeros((), device=grads[0].device, dtype=norm_dtype)
        for grad in grads:
            total_squared_norm.add_(grad.to(norm_dtype).square().sum())
        grad_l2 = total_squared_norm.sqrt()

        if grad_l2 > self.max_l2_norm:
            clip_coeff = self.max_l2_norm / (grad_l2 + self.epslion)
            for grad in grads:
                grad.mul_(clip_coeff)
