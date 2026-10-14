import torch
import torch.nn as nn
from cs336_basics.multihead_self_attention_with_rope import MultiHeadSelfAttention
from cs336_basics.RMSnorm import RMSNorm
from cs336_basics.SwiGLU import SwiGLU
from cs336_basics.rope import RoPE

class TransformerBlock(nn.Module):
    """
    TransformerBlock 是Transformer块，它把包含多头注意力机制的一些组件包装在一起，形成一个完整的Transformer块。
    Args:
        d_model (int): 输入的维度，也就是d_model
        n_heads (int): 头的数量
        d_ff (int): 前馈神经网络的维度
        max_seq_len (int): 最大序列长度
        theta (float): 底数超参数
        attn_q_proj_weight (torch.Tensor): 查询的权重
    """
    def __init__(self, d_model:int, n_heads:int, d_ff:int, max_seq_len:int, theta:float,attn_q_proj_weight:torch.Tensor,attn_k_proj_weight:torch.Tensor,attn_v_proj_weight:torch.Tensor,attn_o_proj_weight:torch.Tensor,ln1_weight:torch.Tensor,ln2_weight:torch.Tensor,ffn_w1_weight:torch.Tensor,ffn_w2_weight:torch.Tensor,ffn_w3_weight:torch.Tensor,device=None):
        super(TransformerBlock, self).__init__()
        if device is None:
            device = attn_q_proj_weight.device
        "权重"
        self.attn_q_proj_weight = nn.Parameter(attn_q_proj_weight.to(device=device).clone())
        self.attn_k_proj_weight = nn.Parameter(attn_k_proj_weight.to(device=device).clone())
        self.attn_v_proj_weight = nn.Parameter(attn_v_proj_weight.to(device=device).clone())
        self.attn_o_proj_weight = nn.Parameter(attn_o_proj_weight.to(device=device).clone())

        self.ln1_weight = ln1_weight
        self.ln2_weight = ln2_weight

        self.ffn_w1_weight = ffn_w1_weight
        self.ffn_w2_weight = ffn_w2_weight
        self.ffn_w3_weight = ffn_w3_weight
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        # self.linear_module = LinearModule(d_model,d_model,device)
        # self.embedding_module = EmbeddingModule(d_model,d_model,device)
        self.rms_norm1 = RMSNorm(d_model,eps=1e-5,device=device,dtype=ln1_weight.dtype)
        self.rms_norm1.load_state_dict({"weight":self.ln1_weight})
        self.rms_norm2 = RMSNorm(d_model,eps=1e-5,device=device,dtype=ln2_weight.dtype)
        self.rms_norm2.load_state_dict({"weight":self.ln2_weight})
        self.swiglu = SwiGLU(d_model, d_ff).to(device=device,dtype=ffn_w1_weight.dtype)
        self.swiglu.load_state_dict({"w1.W":self.ffn_w1_weight,"w2.W":self.ffn_w2_weight,"w3.W":self.ffn_w3_weight})
        self.multi_head_self_attention = MultiHeadSelfAttention(d_model,n_heads,theta,max_seq_len,device)

    def forward(self,in_features:torch.Tensor):
        token_positions = torch.arange(in_features.shape[-2],device=in_features.device)
        x1 = self.rms_norm1(in_features)
        x1 = self.multi_head_self_attention(x1,self.attn_q_proj_weight,self.attn_k_proj_weight,self.attn_v_proj_weight,self.attn_o_proj_weight,token_positions)
        x1 = x1 + in_features
        x2 = self.rms_norm2(x1)
        x2 = self.swiglu(x2)
        out = x2 + x1
        return out
