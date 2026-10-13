import pytest
import torch

from cs336_basics.multihead_self_attention_with_rope import MultiHeadSelfAttention


@pytest.mark.parametrize("input_shape", [(5, 8), (3, 5, 8), (2, 3, 5, 8)])
def test_default_positions_match_explicit_positions(input_shape):
    torch.manual_seed(42)
    model = MultiHeadSelfAttention(8, 2, 10000, 32, device="cpu")
    x = torch.randn(input_shape)
    weights = [torch.randn(8, 8) for _ in range(4)]

    actual = model(x, *weights)
    expected = model(x, *weights, token_positions=torch.arange(5))

    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("batch_shape", [(3,), (2, 3)])
def test_batched_positions_match_independent_sequences(batch_shape):
    torch.manual_seed(42)
    model = MultiHeadSelfAttention(8, 2, 10000, 32, device="cpu")
    x = torch.randn(*batch_shape, 5, 8)
    weights = [torch.randn(8, 8) for _ in range(4)]
    offsets = torch.arange(x.numel() // (5 * 8)).reshape(batch_shape)
    positions = offsets.unsqueeze(-1) + torch.arange(5)

    actual = model(x, *weights, token_positions=positions)
    expected = torch.stack([
        model(sequence, *weights, token_positions=sequence_positions)
        for sequence, sequence_positions in zip(
            x.reshape(-1, 5, 8), positions.reshape(-1, 5)
        )
    ]).reshape_as(x)

    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected)
