import json
import math
from argparse import ArgumentParser, Namespace
from pathlib import Path
import torch
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train import build_model
from cs336_basics.softmax import softmax


@torch.no_grad()
def sample_next_token(last_logits, temperature=1.0, top_k=None, top_p=1.0):
    """Sample one token per row of a (batch_size, vocab_size) tensor."""
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive or None")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if temperature == 0:
        return last_logits.argmax(dim=-1, keepdim=True)

    scaled_logits = last_logits / temperature
    if top_k is not None:
        k = min(top_k, scaled_logits.shape[-1])
        values, indices = torch.topk(scaled_logits, k, dim=-1)
        filtered_logits = torch.full_like(scaled_logits, float("-inf"))
        scaled_logits = filtered_logits.scatter(-1, indices, values)

    probabilities = softmax(scaled_logits, dim=-1)
    if top_p < 1:
        sorted_probabilities, sorted_indices = torch.sort(probabilities, dim=-1, descending=True)
        cumulative_probabilities = sorted_probabilities.cumsum(dim=-1)
        remove_mask = cumulative_probabilities >= top_p
        # Shift the mask to retain the token that first reaches the threshold.
        remove_mask[..., 1:] = remove_mask[..., :-1].clone()
        remove_mask[..., 0] = False
        sorted_probabilities = sorted_probabilities.masked_fill(remove_mask, 0)
        sorted_probabilities /= sorted_probabilities.sum(dim=-1, keepdim=True)
        sorted_position = torch.multinomial(sorted_probabilities, num_samples=1)
        return sorted_indices.gather(dim=-1, index=sorted_position)
    return torch.multinomial(probabilities, num_samples=1)


@torch.no_grad()
def generate_tokens(model, input_ids, max_new_tokens, temperature=1.0, top_k=None, eos_token_id=None, top_p=1.0):
    """Return the prompt followed by generated tokens for a single sequence."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must contain one non-empty sequence")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    was_training = model.training
    model.eval()
    generated_ids = input_ids
    try:
        for _ in range(max_new_tokens):
            context_ids = generated_ids[:, -model.context_length:]
            logits = model(context_ids)
            next_token = sample_next_token(logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
    finally:
        model.train(was_training)
    return generated_ids


def main(argv=None):
    project_dir = Path(__file__).resolve().parent.parent
    parser = ArgumentParser(description="Generate text from a trained Transformer checkpoint")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 selects the highest-scoring token")
    parser.add_argument("--top-k", type=int, default=50, help="Number of candidates to keep; 0 disables filtering")
    parser.add_argument("--top-p", type=float, default=1.0, help="Cumulative probability threshold; 1 disables filtering")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=Path, default=project_dir / "checkpoints/tinystories/final.pt")
    parser.add_argument("--config", type=Path, help="Training config JSON; inferred when the checkpoint directory has one")
    parser.add_argument("--vocab", type=Path, default=project_dir / "tinystories_vocab.pkl")
    parser.add_argument("--merges", type=Path, default=project_dir / "tinystories_merges.pkl")
    parser.add_argument("--output", type=Path, help="Also save generated text to this file")
    generation_args = parser.parse_args(argv)
    if not math.isfinite(generation_args.temperature) or generation_args.temperature < 0:
        parser.error("temperature must be finite and non-negative")
    if generation_args.top_k < 0:
        parser.error("top-k must be non-negative")
    if not 0 < generation_args.top_p <= 1:
        parser.error("top-p must be in (0, 1]")
    if generation_args.max_new_tokens < 0:
        parser.error("max-new-tokens must be non-negative")

    device = torch.device(generation_args.device)
    checkpoint_path = generation_args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    print(checkpoint.keys())
    print("训练步数：", checkpoint["iteration"])

    config_path = generation_args.config
    if config_path is None:
        configs = list(checkpoint_path.parent.glob("config_*.json"))
        if len(configs) != 1:
            parser.error("provide --config when the checkpoint directory has zero or multiple config files")
        config_path = configs[0]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    args = Namespace(**config)
    model = build_model(args, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded; evaluation mode: {not model.training}")

    tokenizer = Tokenizer.from_files(
        vocab_filepath=generation_args.vocab,
        merges_filepath=generation_args.merges,
        special_tokens=["<|endoftext|>"],
    )

    prompt = generation_args.prompt
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        parser.error("prompt must not be empty")
    if max(tokenizer.vocab) >= model.vocab_size:
        parser.error("tokenizer IDs exceed the model vocabulary; use the tokenizer used during training")
    decoded = tokenizer.decode(token_ids)
    if decoded != prompt:
        raise ValueError("Decoded text does not match the prompt")

    print(f"Token IDs: {token_ids}")
    print(f"Decoded text: {decoded}")

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    print(f"Input shape: {tuple(input_ids.shape)}; dtype: {input_ids.dtype}")

    torch.manual_seed(generation_args.seed)
    generated_ids = generate_tokens(
        model,
        input_ids,
        max_new_tokens=generation_args.max_new_tokens,
        temperature=generation_args.temperature,
        top_k=generation_args.top_k or None,
        top_p=generation_args.top_p,
        eos_token_id=tokenizer.bytes_to_id[b"<|endoftext|>"],
    )
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    print(f"Temperature: {generation_args.temperature}; top-k: {generation_args.top_k}; top-p: {generation_args.top_p}")
    print(f"Generated tokens: {generated_ids.shape[-1] - input_ids.shape[-1]}")
    print(generated_text)
    if generation_args.output is not None:
        generation_args.output.parent.mkdir(parents=True, exist_ok=True)
        generation_args.output.write_text(generated_text, encoding="utf-8")
        print(f"Text saved: {generation_args.output}")


if __name__ == "__main__":
    main()
