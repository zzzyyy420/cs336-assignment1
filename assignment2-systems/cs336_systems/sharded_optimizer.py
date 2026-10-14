from collections.abc import Iterable
from typing import Any

import torch
import torch.distributed as dist


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        optimizer_cls: type[torch.optim.Optimizer],
        **kwargs: Any,
    ) -> None:
        super().__init__(params, kwargs)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        parameter_owners: dict[torch.Tensor, int] = {}
        local_groups = []
        for group in self.param_groups:
            local_params = []
            for parameter in group["params"]:
                if parameter not in parameter_owners:
                    parameter_owners[parameter] = len(parameter_owners) % self.world_size
                if parameter_owners[parameter] == self.rank:
                    local_params.append(parameter)
            local_groups.append({**group, "params": local_params})

        self.parameter_owners = parameter_owners
        self.local_optimizer = optimizer_cls(local_groups, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.local_optimizer.step(closure)
        for parameter, owner in self.parameter_owners.items():
            dist.broadcast(parameter.data, src=owner)
        return loss

    def state_dict(self) -> dict[str, Any]:
        return self.local_optimizer.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.local_optimizer.load_state_dict(state_dict)
