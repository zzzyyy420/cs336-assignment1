import torch
class CrossEntropyLoss:
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def forward(self):
        # 每一行减去最大值，避免 exp 溢出
        logits = self.inputs - self.inputs.max(dim=-1, keepdim=True).values

        # log(sum(exp(logits)))
        log_normalizer = torch.log(torch.exp(logits).sum(dim=-1))

        # 取出每个位置对应的正确 token 的 logit
        target_logits = logits.gather(
            dim=-1,
            index=self.targets.unsqueeze(-1),
        ).squeeze(-1)

        return (log_normalizer - target_logits).mean()
