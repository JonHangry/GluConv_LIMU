import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class PConv(nn.Module):
    def __init__(self, d_model, n_heads, o_model):
        super(PConv, self).__init__()
        self.dim_conv = d_model // n_heads
        self.res_conv = d_model - self.dim_conv
        self.p_conv = nn.Conv1d(self.dim_conv, self.dim_conv, kernel_size=3, padding=1, bias=False)
        self.conv = nn.Conv1d(d_model, o_model, kernel_size=1)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        dim, res = torch.split(x, [self.dim_conv, self.res_conv], dim=1)
        dim = self.p_conv(dim)
        x = torch.cat((dim, res), dim=1)
        x = self.conv(x)
        return x.permute(0, 2, 1)


class DWConv(nn.Module):
    def __init__(self, d_model, o_model):
        super(DWConv, self).__init__()
        self.depth_conv = nn.Conv1d(d_model, d_model, kernel_size=3, stride=1, padding=1, groups=d_model)
        self.point_conv = nn.Conv1d(d_model, o_model, kernel_size=1, stride=1, padding=0, groups=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.depth_conv(x)
        x = self.point_conv(x)

        return x.permute(0, 2, 1)


class ConvFFN(nn.Module):
    def __init__(self, d_model, n_heads, seq_len, o_model, seq_model):
        super(ConvFFN, self).__init__()
        self.LN = nn.LayerNorm(d_model)
        self.PConv = PConv(d_model, n_heads, o_model)

        self.SeqLinear = nn.Linear(seq_len, seq_model)
        self.SeqLinear_Recover = nn.Linear(seq_model, seq_len)

        self.Linear = nn.Linear(o_model, d_model)

        self.DWConv = DWConv(d_model//2, d_model//2)

        self.DWLinear = nn.Linear(d_model//2, d_model)

    def forward(self, x):
        res = x.clone()
        x = self.LN(x)
        x = self.PConv(x)
        x = self.Linear(x)
        x = self.SeqLinear(x.permute(0, 2, 1)).permute(0, 2, 1)

        x1, x2 = x.chunk(2, dim=-1)
        x2 = self.DWConv(x2)

        x = x1 * x2
        x = self.DWLinear(x)
        x = self.SeqLinear_Recover(x.permute(0, 2, 1)).permute(0, 2, 1)

        return x + res

class SparseSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, seq_len, seq_model, dropout=0.1):
        super(SparseSelfAttention, self).__init__()
        self.p_model = d_model // n_heads
        self.n_heads = n_heads
        self.d_model = d_model
        self.Linear = nn.Linear(self.d_model, 3 * self.n_heads * self.p_model)
        self.Linear_Recover = nn.Linear(self.n_heads * self.p_model, d_model)

        self.dropout = nn.Dropout(dropout)

        self.seq_model = seq_model
        self.SeqLinear = nn.Linear(seq_len, self.seq_model)
        self.SeqLinear_Recover = nn.Linear(self.seq_model, seq_len)


    def forward(self, input):
        res = input.clone()
        bs, l, nh, pm = input.size(0), input.size(1), self.n_heads, self.p_model
        qkv = self.Linear(input)
        qkv = self.SeqLinear(qkv.permute(0, 2, 1)).permute(0, 2, 1)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = map(lambda x: x.reshape(bs, -1, nh, pm), [q, k, v])

        scores = torch.einsum('bnhd,bmhd->bhnm', (q,k))

        scores_weight = F.softmax(scores / (self.p_model ** 0.5), dim=-1)
        scores_weight = self.dropout(scores_weight)

        attention = torch.einsum('bhnm,bmhd->bnhd', (scores_weight, v))
        attention = attention.reshape(bs, -1, nh*pm)
        attention = self.Linear_Recover(attention)
        attention = self.SeqLinear_Recover(attention.permute(0, 2, 1)).permute(0, 2, 1)

        return res + attention


class SparseCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, seq_len, seq_model, dropout=0.1):
        super(SparseCrossAttention, self).__init__()
        self.p_model = d_model // n_heads
        self.n_heads = n_heads
        self.d_model = d_model
        self.Linear = nn.Linear(self.d_model, 3 * self.n_heads * self.p_model)
        self.Linear_Recover = nn.Linear(self.n_heads * self.p_model, d_model)

        self.dropout = nn.Dropout(dropout)

        self.seq_model = seq_model
        self.SeqLinear = nn.Linear(seq_len, self.seq_model)
        self.SeqLinear_Recover = nn.Linear(self.seq_model, seq_len)

    def forward(self, input):
        res = input.clone()
        bs, l, nh, pm = input.size(0), input.size(1), self.n_heads, self.p_model
        qkv = self.Linear(input)
        qkv = self.SeqLinear(qkv.permute(0, 2, 1)).permute(0, 2, 1)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = map(lambda x: x.reshape(bs, -1, nh, pm), [q, k, v])

        scores = torch.einsum('bnhd,bmhd->bhnm', (q, k))

        scores_weight = F.softmax(scores / (self.p_model ** 0.5), dim=-1)
        scores_weight = self.dropout(scores_weight)
        attention = torch.einsum('bhnm,bmhd->bnhd', (scores_weight, v))
        attention = attention.reshape(bs, -1, nh * pm)
        attention = self.Linear_Recover(attention)
        attention = self.SeqLinear_Recover(attention.permute(0, 2, 1)).permute(0, 2, 1)

        return res + attention  # 残差连接


class Encoder(nn.Module):
    def __init__(self, e_layers, d_model, n_heads, seq_len, o_model, seq_model):
        super(Encoder, self).__init__()
        self.e_layers = e_layers
        self.ConvFFNs = nn.ModuleList(
            [ConvFFN(d_model, n_heads, seq_len, o_model, seq_model) for i in range(e_layers)]
        )
        self.Conv = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x):
        for ConvFFN in self.ConvFFNs:
            x = ConvFFN(x)
        x = self.Conv(x.permute(0, 2, 1)).permute(0, 2, 1)

        return x

class Necker(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, seq_len, seq_model, o_model, dropout=0.1):
        super(Necker, self).__init__()
        self.BottleNecks = nn.ModuleList(
            [nn.Sequential(
                ConvFFN(d_model, n_heads, seq_len, o_model, seq_model)
            )for i in range(n_layers)])
        self.Conv = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x):
        for BottleNeck in self.BottleNecks:
            x = BottleNeck(x)
        x = self.Conv(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, L, N]

        return x

class Decoder(nn.Module):
    def __init__(self, d_layers, d_model, n_heads, seq_len, seq_model, o_model, dropout=0.1):
        super(Decoder, self).__init__()
        self.Coders = nn.ModuleList(
            [nn.Sequential(
                ConvFFN(d_model, n_heads, seq_len, o_model, seq_model)
            ) for i in range(d_layers)])
        self.Conv = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x):
        for Coder in self.Coders:
            x = Coder(x)
        x = self.Conv(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, L, N]

        return x

class ConvAll(nn.Module):
    def __init__(self, layers, e_layers, n_layers, d_layers, d_model, n_heads, seq_len, seq_model, o_model, dropout=0.1):
        super(ConvAll, self).__init__()
        self.layers = layers
        self.Conv = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.Encoders = nn.ModuleList(
            [Encoder(e_layers, d_model, n_heads, seq_len, o_model, seq_model) for i in range(layers)]
        )
        self.Neckers = Necker(n_layers, d_model, n_heads, seq_len, seq_model, o_model, dropout)
        self.Decoders = nn.ModuleList(
            [Decoder(d_layers, d_model, n_heads, seq_len, seq_model, o_model, dropout) for i in range(layers)]
        )
        self.Conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)

    def forward(self, x):
        Res = x.clone()
        x = self.Conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        for i in range(self.layers):
            x = self.Encoders[i](x)
            res = x.clone()
            x = self.Neckers(x)
            x = self.Decoders[i](x) + res
        x = self.Conv2(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x + Res