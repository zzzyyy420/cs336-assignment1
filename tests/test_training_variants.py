import torch

from cs336_basics.train import PostNormBlock, build_model, build_parser


def config(*options):
    return build_parser().parse_args([
        "--vocab-size", "32", "--context-length", "8", "--d-model", "16",
        "--d-ff", "32", "--num-layers", "1", "--num-heads", "2", *options,
    ])


def test_postnorm_uses_residual_then_norm():
    model = build_model(config("--norm-position", "post"), "cpu")
    wrapper = model.transformer_blocks[0]
    assert isinstance(wrapper, PostNormBlock)
    block = wrapper.block
    x = torch.randn(2, 5, 16)
    positions = torch.arange(5)
    attention = block.multi_head_self_attention(
        x, block.attn_q_proj_weight, block.attn_k_proj_weight,
        block.attn_v_proj_weight, block.attn_o_proj_weight, positions,
    )
    residual = block.rms_norm1(x + attention)
    expected = block.rms_norm2(residual + block.swiglu(residual))
    torch.testing.assert_close(wrapper(x), expected)


def test_nope_is_invariant_to_supplied_positions():
    model = build_model(config("--no-rope"), "cpu")
    block = model.transformer_blocks[0]
    x = torch.randn(2, 5, 16)
    weights = [block.attn_q_proj_weight, block.attn_k_proj_weight, block.attn_v_proj_weight, block.attn_o_proj_weight]
    a = block.multi_head_self_attention(x, *weights, torch.arange(5))
    b = block.multi_head_self_attention(x, *weights, torch.arange(5) + 20)
    torch.testing.assert_close(a, b)


def test_silu_has_two_trainable_matrices_and_backpropagates():
    model = build_model(config("--ffn-type", "silu"), "cpu")
    ffn = model.transformer_blocks[0].swiglu
    assert sum(p.numel() for p in ffn.parameters()) == 8 * 16**2
    model(torch.tensor([[1, 2, 3]])).square().mean().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_tied_embeddings_share_storage_and_survive_state_loading():
    args = config("--tie-embeddings")
    model = build_model(args, "cpu")
    assert model.linear_module.W is model.embedding_module.embedding_matrix
    model(torch.tensor([[1, 2, 3]])).sum().backward()
    assert model.linear_module.W.grad is not None
    restored = build_model(args, "cpu")
    restored.load_state_dict(model.state_dict())
    assert restored.linear_module.W is restored.embedding_module.embedding_matrix
    torch.testing.assert_close(model(torch.tensor([[1, 2, 3]])), restored(torch.tensor([[1, 2, 3]])))
