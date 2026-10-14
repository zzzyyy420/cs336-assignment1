from types import MethodType

import torch
import torch.distributed as dist
import torch.nn.functional as F
from cs336_basics.model import Embedding, Linear


def _gather_weight(shard: torch.Tensor, shape: torch.Size, compute_dtype: torch.dtype | None) -> torch.Tensor:
    communicated_shard = shard.to(compute_dtype) if compute_dtype is not None else shard
    gathered = [torch.empty_like(communicated_shard) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, communicated_shard.contiguous())
    return torch.cat(gathered)[: shape.numel()].view(shape)


def _reduce_gradient(gradient: torch.Tensor, shard: torch.Tensor) -> torch.Tensor:
    flat = gradient.flatten().to(shard.dtype)
    padded = F.pad(flat, (0, shard.numel() * dist.get_world_size() - flat.numel())).contiguous()
    reduced = torch.empty_like(shard)
    dist.reduce_scatter_tensor(reduced, padded, op=dist.ReduceOp.SUM)
    return reduced / dist.get_world_size()


class _ShardedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, shard, shape, compute_dtype):
        weight = _gather_weight(shard, shape, compute_dtype)
        ctx.shape = shape
        ctx.compute_dtype = compute_dtype
        ctx.save_for_backward(inputs, shard)
        return F.linear(inputs, weight)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, shard = ctx.saved_tensors
        weight = _gather_weight(shard, ctx.shape, ctx.compute_dtype)
        grad_input = grad_output @ weight
        grad_weight = grad_output.reshape(-1, ctx.shape[0]).T @ inputs.reshape(-1, ctx.shape[1])
        grad_shard = _reduce_gradient(grad_weight, shard)
        return grad_input, grad_shard, None, None


class _ShardedEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, token_ids, shard, shape, compute_dtype):
        weight = _gather_weight(shard, shape, compute_dtype)
        ctx.shape = shape
        ctx.save_for_backward(token_ids, shard)
        return F.embedding(token_ids, weight)

    @staticmethod
    def backward(ctx, grad_output):
        token_ids, shard = ctx.saved_tensors
        grad_weight = torch.zeros(ctx.shape, device=grad_output.device, dtype=grad_output.dtype)
        grad_weight.index_add_(0, token_ids.flatten(), grad_output.reshape(-1, ctx.shape[1]))
        grad_shard = _reduce_gradient(grad_weight, shard)
        return None, grad_shard, None, None


def _linear_forward(module: Linear, inputs: torch.Tensor) -> torch.Tensor:
    return _ShardedLinear.apply(inputs, module.weight, module._fsdp_weight_shape, module._fsdp_compute_dtype)


def _embedding_forward(module: Embedding, token_ids: torch.Tensor) -> torch.Tensor:
    return _ShardedEmbedding.apply(token_ids, module.weight, module._fsdp_weight_shape, module._fsdp_compute_dtype)


class FSDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None) -> None:
        super().__init__()
        if not dist.is_initialized():
            raise RuntimeError("FSDP requires an initialized process group")
        self.module = module
        self.compute_dtype = compute_dtype
        self.sharded_weight_names: set[str] = set()
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        for name, child in module.named_modules():
            if not isinstance(child, (Linear, Embedding)):
                continue
            weight = child.weight.detach().clone()
            dist.broadcast(weight, src=0)
            flat = weight.flatten()
            shard_size = (flat.numel() + world_size - 1) // world_size
            padded = F.pad(flat, (0, shard_size * world_size - flat.numel()))
            child.weight = torch.nn.Parameter(padded[rank * shard_size : (rank + 1) * shard_size].clone())
            child._fsdp_weight_shape = weight.shape
            child._fsdp_compute_dtype = compute_dtype
            child.forward = MethodType(_linear_forward if isinstance(child, Linear) else _embedding_forward, child)
            self.sharded_weight_names.add(f"{name}.weight" if name else "weight")

        for name, parameter in module.named_parameters():
            if name in self.sharded_weight_names:
                continue
            dist.broadcast(parameter.data, src=0)
            if parameter.requires_grad:
                parameter.register_hook(self._average_gradient)

    @staticmethod
    def _average_gradient(gradient: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(gradient)
        return gradient / dist.get_world_size()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        pass

    @torch.no_grad()
    def gather_full_params(self) -> dict[str, torch.Tensor]:
        parameters = {}
        for name, parameter in self.module.named_parameters():
            if name in self.sharded_weight_names:
                module_name = name.rpartition(".")[0]
                child = self.module.get_submodule(module_name) if module_name else self.module
                parameters[name] = _gather_weight(parameter.data, child._fsdp_weight_shape, None)
            else:
                parameters[name] = parameter.data.clone()
        return parameters
