import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from lib.GluConcern.attention import MultiheadAttention

def activation_selection(activation):
    if activation == 'SIGMOID':
        return nn.Sigmoid()
    elif activation == 'TANH':
        return nn.Tanh()
    elif activation == 'RELU':
        return nn.ReLU()
    elif activation == 'GELU':
        return nn.GELU()
    elif activation == 'SILU':
        return nn.SiLU()
    elif activation == 'LEAKYRELU':
        return nn.LeakyReLU()
    elif activation == 'HARDSWISH':
        return nn.Hardswish()
    else:
        return nn.GELU()

class MambaVote(nn.Module):
    def __init__(self, d_model:int, m_state:int, m_conv:int, m_expand:int):
        super(MambaVote, self).__init__()
        self.forward_mamba = Mamba(
            d_model=d_model,
            d_state=m_state,
            d_conv=m_conv,
            expand=m_expand,
        )
        self.backward_mamba = Mamba(
            d_model=d_model,
            d_state=m_state,
            d_conv=m_conv,
            expand=m_expand,
        )

    def forward(self, g_data):
        # g_data [B, N, L]
        model_dependency = self.forward_mamba(g_data) + self.backward_mamba(g_data.flip(dims=[-1])).flip(dims=[-1]) # [B, N, L]
        # model_dependency = self.forward_mamba(g_data) # [B, N, L]
        model_dependency = model_dependency + g_data # [B, N, L]

        return model_dependency



class EncoderConcern(nn.Module):
    def __init__(self, attention, n_heads, d_model:int, d_state:int, d_conv:int, expand:int, m_state: int, m_conv: int, m_expand: int, activation:str, dropout):
        super(EncoderConcern, self).__init__()
        # self.Decompose = Mamba(
        #     d_model=d_model,
        #     d_state=d_state,
        #     d_conv=d_conv,
        #     expand=expand
        # )
        self.trend_dep = MambaVote(d_model, m_state, m_conv, m_expand)
        self.encode_attn = MultiheadAttention(d_model=d_model, n_heads=n_heads, d_keys=d_model // n_heads, mask_flag=attention, r_att_drop=dropout)
        #
        # self.another_dep = MambaVote(d_model, m_state, m_conv, m_expand)
        self.activation = activation_selection(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g_data):
        # g_data [B, L, N]
        trend_data = self.trend_dep(g_data)
        attention_vote = self.encode_attn(trend_data, trend_data, trend_data)  # [B, L, N]

        # trend_data = self.Decompose(g_data)  # [B, L, N]

        # other_data = g_data - trend_data
        # other_data = self.another_dep(other_data)
        #
        # trend_vote = other_data + trend_data

        to_vote = attention_vote  # [B, L, N]

        # another_signal = g_data - trend_signal - other_vote  # [B, L, N]
        # another_vote = self.another_dep(another_signal)  # [B, L, N]
        #
        # all_vote = another_vote + to_vote
        all_vote = to_vote
        all_vote = self.activation(all_vote)  # [B, L, N]
        all_vote = self.dropout(all_vote)  # [B, L, N]

        all_signal = all_vote

        return all_signal  # [B, L, N]