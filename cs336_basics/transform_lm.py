import torch
import torch.nn as nn
from cs336_basics.TransformerBlock import TransformerBlock
from cs336_basics.linear import LinearModule
from cs336_basics.embedding import EmbeddingModule
from cs336_basics.RMSnorm import RMSNorm

class TransformerLM(nn.Module):

    def __init__(self, vocab_size:int, context_length:int, d_model:int, num_layers:int, num_heads:int, d_ff:int, rope_theta:float, weights:dict[str, torch.Tensor]):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.weights = weights
        self.embedding_module = EmbeddingModule(vocab_size,d_model,device=None)
        self.embedding_module.load_state_dict({"embedding_matrix":self.weights["token_embeddings.weight"]})
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model,num_heads,d_ff,context_length,rope_theta,self.weights[f"layers.{layer}.attn.q_proj.weight"],self.weights[f"layers.{layer}.attn.k_proj.weight"],self.weights[f"layers.{layer}.attn.v_proj.weight"],self.weights[f"layers.{layer}.attn.output_proj.weight"],self.weights[f"layers.{layer}.ln1.weight"],self.weights[f"layers.{layer}.ln2.weight"],self.weights[f"layers.{layer}.ffn.w1.weight"],self.weights[f"layers.{layer}.ffn.w2.weight"],self.weights[f"layers.{layer}.ffn.w3.weight"],device=None) for layer in range(num_layers)])
        self.rms_norm = RMSNorm(d_model,eps=1e-5,device=None)
        self.rms_norm.load_state_dict({"weight":self.weights["ln_final.weight"]})
        self.linear_module = LinearModule(d_model,vocab_size,device=None)
        self.linear_module.load_state_dict({"W":self.weights["lm_head.weight"]})
    def forward(self, in_indices):
        embedding = self.embedding_module(in_indices)
        for block in self.transformer_blocks:
            embedding = block(embedding)

        transformer_block_out = embedding   
        out_norm = self.rms_norm(transformer_block_out)

        out_linear = self.linear_module(out_norm)
        return out_linear





