import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.cuda.nvtx as nvtx
import wandb
from cs336_systems.profiling_data import load_token_ids, training_batch, validation_batches
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import clip_gradient, cross_entropy
from cs336_basics.optimizer import AdamW

timestamp = time.strftime("%Y%m%d_%H%M%S")


def profile_range(name: str, device: torch.device):
    return nvtx.range(name) if device.type == "cuda" else nullcontext()


def wandb_init(d_model, d_ff, n_layers, n_heads, batch_size, epochs, train_steps):
    wandb.login()
    run = wandb.init(project="cs336_final_train",
                    config = {
                    # Experiment
                    "experiment_name": f"tinystories_17M_{timestamp}",
                    "total_tokens_processed": 327_680_000,

                    # Data
                    "train_data_path": "../data/TinyStoriesV2-GPT4-train.txt",
                    "valid_data_path": "../data/TinyStoriesV2-GPT4-valid.txt",
                    "encoded_ids_train_path": "./cs336_systems/encoded_ids_train.pkl",
                    "encoded_ids_valid_path": "./cs336_systems/encoded_ids_valid.pkl",
                    "vocab_path": "vocab.json",
                    "merges_path": "merges.txt",

                    # Model
                    "vocab_size": 10000,
                    "context_length": 256,
                    "d_model": d_model,
                    "d_ff": d_ff,
                    "n_layers": n_layers,
                    "n_heads": n_heads,
                    "rope_theta": 10000.0,

                    # Training
                    "batch_size": batch_size, # Adjust based on your GPU memory
                    # "learning_rate": 3e-5,
                    #学习率退火相关参数
                    "initial_lr": 3e-5,
                    "max_learning_rate": 3e-5,
                    "min_learning_rate": 1e-5,
                    "lr_warmup_steps": 1000,
                    "cosine_cycle_iters": 10000,

                    #优化器相关参数
                    "weight_decay": 0.1,
                    "adam_beta1": 0.9,
                    "adam_beta2": 0.95,
                    "eps": 1e-8,

                    #梯度裁剪
                    "grad_clip": 1.0,

                    #训练相关参数
                    "epochs": epochs,
                    "train_steps": train_steps,

                    # Logging & Checkpointing
                    "log_interval": 20,
                    "val_interval": 20,
                    "checkpoint_interval": 60,
                    "checkpoint_dir": "checkpoints",
                }
                )
    config = run.config
    return config


def train(config,device):
    # data_path = config["train_data_path"]
    vocab_size = config["vocab_size"]
    encoded_ids_train_path = config["encoded_ids_train_path"]
    encoded_ids_valid_path = config["encoded_ids_valid_path"]
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    #直接导入编码后的数据
    train_encode_ids = load_token_ids(encoded_ids_train_path)
    valid_encode_ids = load_token_ids(encoded_ids_valid_path)

    # 加载模型
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["n_layers"],
        num_heads=config["n_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
    ).to(device)
    # 加载优化器
    # lr_scheduler = CosineSchedule(config["max_learning_rate"], config["min_learning_rate"], config["lr_warmup_steps"], config["cosine_cycle_iters"]) # 学习率退火
    optimizer = AdamW(model.parameters(), config["initial_lr"], (config["adam_beta1"], config["adam_beta2"]), config["eps"], config["weight_decay"])
    # 学习率退火
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1, total_iters=config["lr_warmup_steps"]) # 线性预热
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"]*config["train_steps"], eta_min=config["min_learning_rate"]) # 余弦退火
    lr_scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[config["lr_warmup_steps"]]) # 顺序学习率退火，milestones表示在第几个step切换到余弦退火
    # 加载损失函数
    # print("模型加载完成")
    # 训练循环
    model.train()
    global_step = 0  # 初始化全局步数

    for epoch in range(config["epochs"]):#每个epoch代表训练完了一整个所有批次的数据，不能轻易和step混淆
        for step in range(config["train_steps"]):
            # 更新学习率
            # new_lr = lr_scheduler(global_step)
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = new_lr
            # print("epoch",epoch,"lr",new_lr)
            x, y = training_batch(train_encode_ids, config["batch_size"], config["context_length"], device)
            with profile_range("forward", device):
                logits = model(x)  # shape: (batch, seq, vocab_size)
                loss = cross_entropy(logits, y)
                if device.type == 'cuda':
                    torch.cuda.synchronize()

            # loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            optimizer.zero_grad()
            with profile_range("backward", device):
                loss.backward()
                if device.type == 'cuda':
                    torch.cuda.synchronize()

            with profile_range("clip_gradient", device):
                clip_gradient(model.parameters(), config["grad_clip"])

            with profile_range("optimizer_step", device):
                optimizer.step()
            with profile_range("lr_scheduler_step", device):
                lr_scheduler.step()
            global_step += 1 # 增加全局步数

            if step % 100 == 0:
                print(f"Epoch {epoch} Step {step} LR {lr_scheduler.get_last_lr()[0]:.6f} Loss: {loss.item()}")

        # print("训练完了")
        # wandb.log({"epoch": epoch, "loss": loss.item()})
        if (epoch+1) % config["val_interval"] == 0:
            model.eval()
            with torch.no_grad():
                for x, y in validation_batches(
                    valid_encode_ids, config["batch_size"], config["context_length"], device
                ):
                    logits = model(x)
                    #   loss = loss_fn.forward(logits, y)
                    loss = cross_entropy(logits, y)
                    # wandb.log({"epoch": epoch, "loss": loss.item()})
            model.train()
        # print("经过验证集")
        if (epoch+1) % config["checkpoint_interval"] == 0:
            # torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 可以加上'loss': loss.item()等
            }, checkpoint_dir / f"model_epoch_{epoch}_{timestamp}.pth")
            print(f"Checkpoint saved at epoch {epoch}")

    torch.save(model.state_dict(), checkpoint_dir / f"model_final_{timestamp}.pth")
    print("Final checkpoint saved")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:6")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--train_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument('--no-rmsnorm', dest='use_rmsnorm', action='store_false', help="Disable RMSNorm and use LayerNorm instead")

    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=12)


    parser.set_defaults(use_rmsnorm=True)
    args = parser.parse_args()

    device = args.device
    epochs = args.epochs
    train_steps = args.train_steps
    batch_size = args.batch_size
    d_model = args.d_model
    d_ff = args.d_ff
    n_layers = args.n_layers
    n_heads = args.n_heads

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    # config = wandb_init(d_model, d_ff, n_layers, n_heads, batch_size, epochs, train_steps)
    config = {
                    # Experiment
                    "experiment_name": f"tinystories_17M_{timestamp}",
                    "total_tokens_processed": 327_680_000,

                    # Data
                    "train_data_path": "../data/TinyStoriesV2-GPT4-train.txt",
                    "valid_data_path": "../data/TinyStoriesV2-GPT4-valid.txt",
                    "encoded_ids_train_path": "./cs336_systems/encoded_ids_train.pkl",
                    "encoded_ids_valid_path": "./cs336_systems/encoded_ids_valid.pkl",
                    "vocab_path": "vocab.json",
                    "merges_path": "merges.txt",

                    # Model
                    "vocab_size": 10000,
                    "context_length": 256,
                    "d_model": d_model,
                    "d_ff": d_ff,
                    "n_layers": n_layers,
                    "n_heads": n_heads,
                    "rope_theta": 10000.0,

                    # Training
                    "batch_size": batch_size, # Adjust based on your GPU memory
                    # "learning_rate": 3e-5,
                    #学习率退火相关参数
                    "initial_lr": 3e-5,
                    "max_learning_rate": 3e-5,
                    "min_learning_rate": 1e-5,
                    "lr_warmup_steps": 2000,
                    "cosine_cycle_iters": 10000,

                    #优化器相关参数
                    "weight_decay": 0.1,
                    "adam_beta1": 0.9,
                    "adam_beta2": 0.95,
                    "eps": 1e-8,

                    #梯度裁剪
                    "grad_clip": 1.0,

                    #训练相关参数
                    "epochs": epochs,
                    "train_steps": train_steps,

                    # Logging & Checkpointing
                    "log_interval": 20,
                    "val_interval": 20,
                    "checkpoint_interval": 60,
                    "checkpoint_dir": "checkpoints",
                }
    train(config,device)
