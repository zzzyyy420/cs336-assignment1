import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.AdamW import AdamW
from cs336_basics.CheckPoint import save_checkpoint
from cs336_basics.DataLoader import DataLoader
from cs336_basics.GradientClip import GradientClip
from cs336_basics.LrCosineShedule import CosineSchedule
from cs336_basics.cross_entropy import CrossEntropyLoss
from cs336_basics.embedding import EmbeddingModule
from cs336_basics.linear import LinearModule
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transform_lm import TransformerLM


class NoPositionEncoding(torch.nn.Module):
    def forward(self, x, token_positions):
        return x


class SiLUFeedForward(torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w1 = LinearModule(d_model, 4 * d_model)
        self.w2 = LinearModule(4 * d_model, d_model)

    def forward(self, x):
        gate = self.w1(x)
        return self.w2(gate * torch.sigmoid(gate))


class PostNormBlock(torch.nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        block = self.block
        positions = torch.arange(x.shape[-2], device=x.device)
        attention = block.multi_head_self_attention(
            x, block.attn_q_proj_weight, block.attn_k_proj_weight,
            block.attn_v_proj_weight, block.attn_o_proj_weight, positions,
        )
        x = block.rms_norm1(x + attention)
        return block.rms_norm2(x + block.swiglu(x))


def build_model(args, device):
    weights = {
        "token_embeddings.weight": EmbeddingModule(args.vocab_size, args.d_model).embedding_matrix.detach(),
        "ln_final.weight": torch.ones(args.d_model),
        "lm_head.weight": LinearModule(args.d_model, args.vocab_size).W.detach(),
    }
    if getattr(args, "tie_embeddings", False):
        torch.nn.init.trunc_normal_(weights["token_embeddings.weight"], std=0.02, a=-0.06, b=0.06)
    for layer in range(args.num_layers):
        prefix = f"layers.{layer}."
        for name in ("q_proj", "k_proj", "v_proj", "output_proj"):
            weights[prefix + f"attn.{name}.weight"] = LinearModule(args.d_model, args.d_model).W.detach()
        weights[prefix + "ln1.weight"] = torch.ones(args.d_model)
        weights[prefix + "ln2.weight"] = torch.ones(args.d_model)
        weights[prefix + "ffn.w1.weight"] = LinearModule(args.d_model, args.d_ff).W.detach()
        weights[prefix + "ffn.w2.weight"] = LinearModule(args.d_ff, args.d_model).W.detach()
        weights[prefix + "ffn.w3.weight"] = LinearModule(args.d_model, args.d_ff).W.detach()
    model = TransformerLM(
        vocab_size=args.vocab_size, context_length=args.context_length, d_model=args.d_model,
        num_layers=args.num_layers, num_heads=args.num_heads, d_ff=args.d_ff,
        rope_theta=args.rope_theta, weights=weights,
    )
    # The modules have copied the initial weights into their trainable parameters.
    model.weights = None
    for index, block in enumerate(model.transformer_blocks):
        for name in ("ln1_weight", "ln2_weight", "ffn_w1_weight", "ffn_w2_weight", "ffn_w3_weight"):
            setattr(block, name, None)
        if args.no_rmsnorm:
            block.rms_norm1 = torch.nn.Identity()
            block.rms_norm2 = torch.nn.Identity()
        if getattr(args, "no_rope", False):
            block.multi_head_self_attention.rope = NoPositionEncoding()
        if getattr(args, "ffn_type", "swiglu") == "silu":
            block.swiglu = SiLUFeedForward(args.d_model)
        if getattr(args, "norm_position", "pre") == "post":
            model.transformer_blocks[index] = PostNormBlock(block)
    if args.no_rmsnorm:
        model.rms_norm = torch.nn.Identity()
    if getattr(args, "tie_embeddings", False):
        model.linear_module.W = model.embedding_module.embedding_matrix
    return model.to(device)


def prepare_tokens(source, output, tokenizer, chunk_size=1024 * 1024):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("token output must be different from the input text")
    if output.exists():
        raise FileExistsError(f"Token output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype("uint16" if max(tokenizer.vocab) < 65536 else "uint32")
    temporary = output.with_name(output.name + ".tmp")
    marker = "<|endoftext|>"
    count = 0
    try:
        with source.open("r", encoding="utf-8", newline="") as reader, temporary.open("wb") as writer:
            pending = ""
            while text := reader.read(chunk_size):
                pending += text
                # Retain the unfinished document, including a partial special token.
                boundary = pending.rfind(marker)
                if boundary != -1:
                    end = boundary + len(marker)
                    ids = tokenizer.encode(pending[:end])
                    np.asarray(ids, dtype=dtype).tofile(writer)
                    count += len(ids)
                    pending = pending[end:]
                    print(f"Encoded {count:,} tokens from {source.name}", flush=True)
                if len(pending) > 32 * 1024 * 1024:
                    raise ValueError("No document boundary within 32 MiB; expected <|endoftext|> delimiters")
            if pending:
                ids = tokenizer.encode(pending)
                np.asarray(ids, dtype=dtype).tofile(writer)
                count += len(ids)
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    output.with_name(output.name + ".json").write_text(json.dumps({
        "dtype": dtype.name,
        "token_count": count,
        "vocab_size": max(tokenizer.vocab) + 1,
        "source": str(source),
    }, indent=2), encoding="utf-8")
    print(f"Saved {count:,} {dtype.name} tokens to {output}", flush=True)


def load_tokens(path, dtype):
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r", allow_pickle=False)
    if path.suffix == ".pkl":
        with path.open("rb") as source:
            return np.asarray(pickle.load(source))
    metadata = path.with_name(path.name + ".json")
    if metadata.exists():
        dtype = json.loads(metadata.read_text(encoding="utf-8"))["dtype"]
    return np.memmap(path, mode="r", dtype=dtype)


def validate_tokens(data, vocab_size, context_length):
    if data.ndim != 1 or not np.issubdtype(data.dtype, np.integer):
        raise ValueError("token data must be a one-dimensional integer array")
    if len(data) <= context_length:
        raise ValueError("token data must contain at least context_length + 1 tokens")
    for start in range(0, len(data), 1024 * 1024):
        chunk = data[start:start + 1024 * 1024]
        if chunk.min() < 0 or chunk.max() >= vocab_size:
            raise ValueError(f"token IDs must be in [0, {vocab_size}); check the tokenizer and vocab_size")


@torch.no_grad()
def evaluate(model, loader, num_batches, seed, precision="float32"):
    was_training = model.training
    random_state = np.random.get_state()
    model.eval()
    np.random.seed(seed)
    total_loss = 0.0
    total_tokens = 0
    try:
        for _ in range(num_batches):
            x, y = loader.get_train_batch_data()
            with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=precision == "bfloat16"):
                logits = model(x)
            loss = CrossEntropyLoss(logits.float().reshape(-1, model.vocab_size), y.reshape(-1)).forward()
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
    finally:
        np.random.set_state(random_state)
        model.train(was_training)
    return total_loss / total_tokens


def build_parser():
    parser = argparse.ArgumentParser(description="Train the CS336 Transformer language model")
    parser.add_argument("--prepare-data", action="store_true", help="Encode text into token files and exit")
    parser.add_argument("--train-data", type=Path, help="Token array for training, or text with --prepare-data")
    parser.add_argument("--valid-data", type=Path, help="Token array for validation, or text with --prepare-data")
    parser.add_argument("--train-tokens", type=Path, default=Path("data/tinystories_train.bin"))
    parser.add_argument("--valid-tokens", type=Path, default=Path("data/tinystories_valid.bin"))
    parser.add_argument("--vocab", type=Path, default=Path("tinystories_vocab.pkl"))
    parser.add_argument("--merges", type=Path, default=Path("tinystories_merges.pkl"))
    parser.add_argument("--token-dtype", choices=["uint16", "uint32", "int64"], default="uint16")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--no-rmsnorm", action="store_true")
    parser.add_argument("--norm-position", choices=["pre", "post"], default="pre")
    parser.add_argument("--no-rope", action="store_true")
    parser.add_argument("--ffn-type", choices=["swiglu", "silu"], default="swiglu")
    parser.add_argument("--tie-embeddings", action="store_true")
    parser.add_argument("--precision", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--matmul-precision", choices=["highest", "high"], default="highest")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-steps", type=int, default=2000, help="Total desired optimizer steps, including resumed steps")
    parser.add_argument("--max-learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--cosine-cycle-iters", type=int, help="Decay endpoint; defaults to --train-steps")
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--val-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20, help="Fixed random validation batches per evaluation")
    parser.add_argument("--eval-batch-size", type=int, help="Defaults to training batch size; fix across comparisons")
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/tinystories"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", help="Also log metrics to Weights & Biases")
    parser.add_argument("--wandb-project", default="cs336-assignment1")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.prepare_data:
        if args.train_data is None or args.valid_data is None:
            parser.error("--prepare-data requires --train-data and --valid-data text paths")
        tokenizer = Tokenizer.from_files(args.vocab, args.merges, ["<|endoftext|>"])
        prepare_tokens(args.train_data, args.train_tokens, tokenizer)
        prepare_tokens(args.valid_data, args.valid_tokens, tokenizer)
        return

    positive = [
        "vocab_size", "context_length", "d_model", "d_ff", "num_layers", "num_heads",
        "batch_size", "train_steps", "log_interval", "val_interval", "eval_batches", "checkpoint_interval",
    ]
    if any(getattr(args, name) <= 0 for name in positive):
        parser.error("model sizes, step counts, batch sizes, and intervals must be positive")
    if args.d_model % args.num_heads or (args.d_model // args.num_heads) % 2:
        parser.error("d_model must divide into even-sized attention heads")
    decay_end = args.cosine_cycle_iters if args.cosine_cycle_iters is not None else args.train_steps
    if not 0 <= args.warmup_steps < decay_end:
        parser.error("require 0 <= warmup_steps < cosine_cycle_iters")
    if not 0 <= args.min_learning_rate <= args.max_learning_rate:
        parser.error("require 0 <= min_learning_rate <= max_learning_rate")
    if not (0 <= args.beta1 < 1 and 0 <= args.beta2 < 1) or args.eps <= 0 or args.weight_decay < 0:
        parser.error("invalid AdamW hyperparameters")
    if args.grad_clip <= 0:
        parser.error("grad_clip must be positive")
    if args.eval_batch_size is not None and args.eval_batch_size <= 0:
        parser.error("eval-batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and (
        not torch.cuda.is_available() or (device.index is not None and device.index >= torch.cuda.device_count())
    ):
        parser.error("requested CUDA device is unavailable; use cpu or an existing GPU such as cuda:0")
    if args.precision == "bfloat16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        parser.error("bfloat16 training requires a supported CUDA GPU")
    if device.type == "cuda":
        torch.set_float32_matmul_precision(args.matmul_precision)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_path = args.train_data or args.train_tokens
    valid_path = args.valid_data or args.valid_tokens
    if not train_path.is_file() or not valid_path.is_file():
        parser.error("token files are missing; run --prepare-data first or provide token array paths")
    train_data = load_tokens(train_path, args.token_dtype)
    valid_data = load_tokens(valid_path, args.token_dtype)
    validate_tokens(train_data, args.vocab_size, args.context_length)
    validate_tokens(valid_data, args.vocab_size, args.context_length)
    train_loader = DataLoader(train_data, args.batch_size, args.context_length, device=device)
    valid_loader = DataLoader(valid_data, args.eval_batch_size or args.batch_size, args.context_length, device=device)
    model = build_model(args, device)
    optimizer = AdamW(model.parameters(), args.max_learning_rate, (args.beta1, args.beta2), args.eps, args.weight_decay)
    schedule = CosineSchedule(args.max_learning_rate, args.min_learning_rate, args.warmup_steps, decay_end)
    clip_gradients = GradientClip(model.parameters(), args.grad_clip)
    global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        global_step = checkpoint["iteration"]
    if global_step >= args.train_steps:
        parser.error("checkpoint already reached train_steps; choose a larger total step count")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["cosine_cycle_iters"] = decay_end
    config["total_tokens_processed"] = args.train_steps * args.batch_size * args.context_length
    config["num_parameters"] = sum(p.numel() for p in model.parameters())
    config["effective_d_ff"] = 4 * args.d_model if args.ffn_type == "silu" else args.d_ff
    config["hardware"] = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    config_path = args.output_dir / f"config_{time.time_ns()}.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Device: {device}; parameters: {config['num_parameters']:,}; starting step: {global_step}", flush=True)
    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, config=config)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        with (args.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as metrics:
            model.train()
            while global_step < args.train_steps:
                lr = schedule(global_step)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad()
                x, y = train_loader.get_train_batch_data()
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.precision == "bfloat16"):
                    logits = model(x)
                loss = CrossEntropyLoss(logits.float().reshape(-1, args.vocab_size), y.reshape(-1)).forward()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite training loss at step {global_step}")
                loss.backward()
                log_due = (global_step + 1) % args.log_interval == 0 or global_step + 1 == args.train_steps
                if log_due:
                    grad_norm = torch.linalg.vector_norm(torch.stack([
                        p.grad.float().norm() for p in model.parameters() if p.grad is not None
                    ])).item()
                    if not np.isfinite(grad_norm):
                        raise RuntimeError(f"Non-finite gradient norm at step {global_step}")
                clip_gradients()
                optimizer.step()
                global_step += 1

                record = {"step": global_step}
                if log_due:
                    record.update(train_loss=loss.item(), learning_rate=lr, gradient_norm_before_clip=grad_norm)
                if global_step % args.val_interval == 0 or global_step == args.train_steps:
                    record["valid_loss"] = evaluate(model, valid_loader, args.eval_batches, args.seed, args.precision)
                    record["valid_tokens"] = args.eval_batches * valid_loader.batch_size * args.context_length
                if len(record) > 1:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                        record["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                    record["elapsed_seconds"] = time.perf_counter() - started
                    record["tokens_processed"] = global_step * args.batch_size * args.context_length
                    metrics.write(json.dumps(record) + "\n")
                    metrics.flush()
                    print(json.dumps(record), flush=True)
                    if run is not None:
                        run.log(record, step=global_step)
                if global_step % args.checkpoint_interval == 0:
                    save_checkpoint(model, optimizer, global_step, args.output_dir / f"step_{global_step}.pt")

            final_path = args.output_dir / "final.pt"
            save_checkpoint(model, optimizer, global_step, final_path)
            print(f"Final checkpoint saved: {final_path}", flush=True)
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
