"""Produce numerical evidence for the assignment's written accounting problems."""

import json
import math
from pathlib import Path

import torch

from cs336_basics.train import build_model, build_parser


ROOT = Path(__file__).resolve().parent.parent


def accounting(vocab_size, seq_len, layers, width, heads, ffn_width=None):
    ffn_width = ffn_width if ffn_width is not None else round((8 * width / 3) / 64) * 64
    parameters = 2 * vocab_size * width + layers * (4 * width**2 + 3 * width * ffn_width + 2 * width) + width
    flops = {
        "qkv_projections": layers * 6 * seq_len * width**2,
        "attention_scores_and_values": layers * 4 * seq_len**2 * width,
        "attention_output_projection": layers * 2 * seq_len * width**2,
        "swiglu_projections": layers * 6 * seq_len * width * ffn_width,
        "lm_head": 2 * seq_len * width * vocab_size,
    }
    total = sum(flops.values())
    # Count one output per listed operation; this is an activation-storage model,
    # not an exact prediction of the autograd allocator's peak memory.
    activation_elements_per_sample = (
        layers * (8 * seq_len * width + 4 * seq_len * ffn_width + 2 * heads * seq_len**2)
        + seq_len * width + 2 * seq_len * vocab_size + seq_len
    )
    return {
        "vocab_size": vocab_size, "context_length": seq_len, "num_layers": layers,
        "d_model": width, "num_heads": heads, "d_ff": ffn_width,
        "parameters": parameters, "parameter_bytes_float32": 4 * parameters,
        "flops": flops, "forward_matmul_flops": total,
        "flop_proportions": {name: value / total for name, value in flops.items()},
        "activation_bytes_per_batch_item_float32": 4 * activation_elements_per_sample,
        "parameter_gradient_optimizer_bytes_float32": 16 * parameters,
        "max_batch_80_decimal_gb_under_output_storage_model": math.floor(
            (80 * 10**9 - 16 * parameters) / (4 * activation_elements_per_sample)
        ),
    }


def main():
    args = build_parser().parse_args([
        "--vocab-size", "32", "--d-model", "16", "--d-ff", "32",
        "--num-layers", "1", "--num-heads", "2", "--context-length", "8",
    ])
    actual = sum(p.numel() for p in build_model(args, "cpu").parameters())
    assert actual == accounting(32, 8, 1, 16, 2, 32)["parameters"]
    torch.manual_seed(42)
    initial = 5 * torch.randn(10, 10)
    toy = {}
    for lr in (10, 100, 1000):
        weights = torch.nn.Parameter(initial.clone())
        optimizer = torch.optim.SGD([weights], lr=lr)
        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            loss = weights.square().mean()
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        toy[str(lr)] = {"losses_before_update": losses, "analytical_loss_multiplier_per_step": (1 - lr / 50)**2}
    models = {
        name: accounting(50257, 1024, layers, width, heads)
        for name, layers, width, heads in (
            ("small", 12, 768, 12), ("medium", 24, 1024, 16),
            ("large", 36, 1280, 20), ("xl", 48, 1600, 25),
        )
    }
    models["xl_context_16384"] = accounting(50257, 16384, 48, 1600, 25)
    xl = models["xl"]
    training_flops = 400000 * (3 * 1024 * xl["forward_matmul_flops"] + 14 * xl["parameters"])
    invalid = {}
    for value in (bytes([0xC3, 0x28]), "\u725b".encode()[:1]):
        try:
            value.decode("utf-8")
        except UnicodeDecodeError as error:
            invalid[repr(value)] = str(error)
    result = {
        "parameter_formula_check": {"actual_small_model": actual, "matches_formula": True},
        "unicode": {
            "chr_zero_repr": repr(chr(0)),
            "example_utf8": list("\u725b".encode()),
            "invalid_sequences": invalid,
        },
        "sgd_learning_rate_tuning": toy,
        "transformer_models": models,
        "activation_storage_assumptions": {
            "per_block": "8*S*D + 4*S*F + 2*H*S*S float32 elements per batch item",
            "rmsnorm_outputs": "2*S*D",
            "qkv_outputs": "3*S*D",
            "score_and_softmax_outputs": "2*H*S*S",
            "weighted_values_and_output_projection": "2*S*D",
            "ffn_w1_w3_silu_product_w2_outputs": "4*S*F + S*D",
            "final_norm_logits_cross_entropy": "S*D + 2*S*V + S",
            "exclusions": "RoPE, residual additions, kernel workspaces, extra autograd intermediates, allocator fragmentation",
            "warning": "80 GB batch capacity is conditional on this simplified bookkeeping model, not measured peak GPU memory",
        },
        "adamw": {
            "flops_per_step_approximate": 14 * xl["parameters"],
            "operations_per_parameter": "m:3, v:4, sqrt/epsilon:2, adaptive update:3, decay:2; group scalars ignored",
            "h100_training_hours_50_percent_mfu": training_flops / (0.5 * 495 * 10**12) / 3600,
            "training_flops": training_flops,
            "d_ff_exact_8_over_3_alternative": accounting(50257, 1024, 48, 1600, 25, 8 * 1600 / 3),
        },
    }
    output = ROOT / "results/assignment_accounting.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Small-model parameter formula verified: {actual}")
    print(f"GPT-2 XL assignment parameters: {xl['parameters']:,}")
    print(f"GPT-2 XL forward matrix FLOPs: {xl['forward_matmul_flops']:,}")
    print(f"Conditional float32 80 GB batch capacity: {xl['max_batch_80_decimal_gb_under_output_storage_model']}")
    print(f"H100 training estimate: {result['adamw']['h100_training_hours_50_percent_mfu']:.2f} hours")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
