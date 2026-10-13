from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    from cs336_basics.linear import LinearModule
    model = LinearModule(
    d_in,
    d_out,
    device=weights.device,
    dtype=weights.dtype,
)
    model.load_state_dict({"W": weights})
    return model(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    from cs336_basics.embedding import EmbeddingModule
    model = EmbeddingModule(
        num_embeddings=vocab_size,
        embedding_dim=d_model,
        device=weights.device,
        dtype=weights.dtype,
    )
    model.load_state_dict({"embedding_matrix": weights})
    return model(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    from cs336_basics.SwiGLU import SwiGLU
    model = SwiGLU(d_model=d_model, d_ff=d_ff)
    model = model.to(
        device=w1_weight.device,
        dtype=w1_weight.dtype,
    )
    model.load_state_dict(
        {
            "w1.W": w1_weight,
            "w2.W": w2_weight,
            "w3.W": w3_weight,
        }
    )
    return model(in_features)



def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    from cs336_basics.scaled_dot_product_attention import ScaledDotProductAttention
    model = ScaledDotProductAttention()
    return model(Q, K, V, mask)



def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    from cs336_basics.multihead_self_attention import MultiHeadSelfAttention
    model = MultiHeadSelfAttention(d_model=d_model, n_heads=num_heads)
    return model(
    in_features,
    wq=q_proj_weight,
    wk=k_proj_weight,
    wv=v_proj_weight,
    wo=o_proj_weight,
    )   


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    from cs336_basics.multihead_self_attention_with_rope import MultiHeadSelfAttention
    model = MultiHeadSelfAttention(d_model=d_model, n_heads=num_heads, theta=theta, max_seq_len=max_seq_len, device=in_features.device)
    return model(in_features, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, token_positions)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    from cs336_basics.rope import RoPE
    model = RoPE(theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=in_query_or_key.device)
    return model(in_query_or_key, token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    from cs336_basics.TransformerBlock import TransformerBlock

    model = TransformerBlock(
        d_model=d_model,
        n_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        theta=theta,
        attn_q_proj_weight=weights["attn.q_proj.weight"],
        attn_k_proj_weight=weights["attn.k_proj.weight"],
        attn_v_proj_weight=weights["attn.v_proj.weight"],
        attn_o_proj_weight=weights["attn.output_proj.weight"],
        ln1_weight=weights["ln1.weight"],
        ln2_weight=weights["ln2.weight"],
        ffn_w1_weight=weights["ffn.w1.weight"],
        ffn_w2_weight=weights["ffn.w2.weight"],
        ffn_w3_weight=weights["ffn.w3.weight"],
        device=in_features.device,
    )
    return model(in_features)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    from cs336_basics.transform_lm import TransformerLM
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        weights=weights,
    )
    return model(in_indices)


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    from cs336_basics.RMSnorm import RMSNorm
    model = RMSNorm(
        d_model=d_model,
        eps=eps,
        device=weights.device,
        dtype=weights.dtype,)
    model.load_state_dict({"weight": weights})
    return model(in_features)

def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    from cs336_basics.SwiGLU import SwiGLU
    return SwiGLU.silu(SwiGLU, in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    from cs336_basics.DataLoader import DataLoader
    loader = DataLoader(dataset, batch_size, context_length, device=device)
    return loader.get_train_batch_data()


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    from cs336_basics.softmax import softmax
    return softmax(in_features, dim=dim)

def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    from cs336_basics.cross_entropy import CrossEntropyLoss
    return CrossEntropyLoss(inputs, targets).forward()


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    from cs336_basics.GradientClip import GradientClip
    GradientClip(parameters, max_l2_norm)()


def get_adamw_cls() -> Any:
    from cs336_basics.AdamW import AdamW
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    from cs336_basics.LrCosineShedule import CosineSchedule
    return CosineSchedule(
        max_learning_rate=max_learning_rate,
        min_learning_rate=min_learning_rate,
        warmup_iters=warmup_iters,
        cosine_cycle_iters=cosine_cycle_iters,
    )(it)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    from cs336_basics.CheckPoint import save_checkpoint
    save_checkpoint(model, optimizer, iteration, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    from cs336_basics.CheckPoint import load_checkpoint
    return load_checkpoint(src, model, optimizer)


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    from cs336_basics.tokenizer import Tokenizer
    return Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    from cs336_basics.bpe import run_train_bpe as train_bpe_impl
    return train_bpe_impl(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        **kwargs,
    )
                        
