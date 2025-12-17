import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

# TODO 进来的输入维度是 [B, L, N]，其中N被扩展到d_model
class PConv(nn.Module):
    def __init__(self, d_model, n_heads, o_model):
        super(PConv, self).__init__()
        self.dim_conv = d_model // n_heads
        self.res_conv = d_model - self.dim_conv
        self.p_conv = nn.Conv1d(self.dim_conv, self.dim_conv, kernel_size=3, padding=1, bias=False)
        self.conv = nn.Conv1d(d_model, o_model, kernel_size=1)
    def forward(self, x):
        # x [B, L, N]
        x = x.permute(0, 2, 1)
        dim, res = torch.split(x, [self.dim_conv, self.res_conv], dim=1)
        dim = self.p_conv(dim)  # TODO 这里可否写12个头分别卷积
        x = torch.cat((dim, res), dim=1)  # [B, N, L]
        x = self.conv(x)  # [B, N, L]
        return x.permute(0, 2, 1)


class DWConv(nn.Module):
    def __init__(self, d_model, o_model):
        super(DWConv, self).__init__()
        self.depth_conv = nn.Conv1d(d_model, d_model, kernel_size=3, stride=1, padding=1, groups=d_model)
        self.point_conv = nn.Conv1d(d_model, o_model, kernel_size=1, stride=1, padding=0, groups=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, N, L]
        x = self.depth_conv(x)  # [B, N, L]
        x = self.point_conv(x)  # [B, N, L]

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
        # x [B, L, N]
        res = x.clone()  # [B, L, N]
        x = self.LN(x)
        x = self.PConv(x)  # [B, L, O]
        x = self.Linear(x)  # [B, L, N]
        x = self.SeqLinear(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, S, N]

        x1, x2 = x.chunk(2, dim=-1)  # [B, S, N//2]
        x2 = self.DWConv(x2)  # [B, S, N//2]

        x = x1 * x2
        x = self.DWLinear(x)  # [B, S, N]
        x = self.SeqLinear_Recover(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, L, N]

        return x + res

# if __name__ == '__main__':
#     data = torch.randn([32, 128, 256])
#     P = ConvFFN(256, 12, 128, 512, 196)
#     data = P(data)
#     print(data.shape)


class SparseSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, seq_len, seq_model, dropout=0.1):
        super(SparseSelfAttention, self).__init__()
        self.p_model = d_model // n_heads  # TODO 计算得到切割后每块的维度
        self.n_heads = n_heads  # TODO 记录每个头数
        self.d_model = d_model  # TODO 记录原本的维度
        self.Linear = nn.Linear(self.d_model, 3 * self.n_heads * self.p_model)  # TODO 规范化到能够被切割整除
        self.Linear_Recover = nn.Linear(self.n_heads * self.p_model, d_model)  # TODO 还原到原本的维度

        self.dropout = nn.Dropout(dropout)

        # 写下对L的处理
        self.seq_model = seq_model
        self.SeqLinear = nn.Linear(seq_len, self.seq_model)
        self.SeqLinear_Recover = nn.Linear(self.seq_model, seq_len)


    def forward(self, input):
        # TODO input:[B, L, d_model]
        res = input.clone()  # TODO 方便后续进行残差连接
        bs, l, nh, pm = input.size(0), input.size(1), self.n_heads, self.p_model
        qkv = self.Linear(input)  # [B, L, 3 * n_heads * p_model]
        qkv = self.SeqLinear(qkv.permute(0, 2, 1)).permute(0, 2, 1)  # [B, S, D]
        q, k, v = qkv.chunk(3, dim=-1)  # Each one: [B, S, n_heads*p_model]=[B, S, D]
        q, k, v = map(lambda x: x.reshape(bs, -1, nh, pm), [q, k, v])  # [B, S, nh, pm]

        scores = torch.einsum('bnhd,bmhd->bhnm', (q,k))  # [B, nh, S, pm]

        scores_weight = F.softmax(scores / (self.p_model ** 0.5), dim=-1)  # [B, nh, S, pm]
        scores_weight = self.dropout(scores_weight)  # [B, nh, S, pm]

        attention = torch.einsum('bhnm,bmhd->bnhd', (scores_weight, v))  # [B, nh, S, pm]
        attention = attention.reshape(bs, -1, nh*pm)  # [B, S, nh*pm]
        attention = self.Linear_Recover(attention)  # [B, S, D]
        attention = self.SeqLinear_Recover(attention.permute(0, 2, 1)).permute(0, 2, 1)  # [B, L, D]

        return res + attention  # 残差连接


class SparseCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, seq_len, seq_model, dropout=0.1):
        super(SparseCrossAttention, self).__init__()
        self.p_model = d_model // n_heads  # TODO 计算得到切割后每块的维度
        self.n_heads = n_heads  # TODO 记录每个头数
        self.d_model = d_model  # TODO 记录原本的维度
        self.Linear = nn.Linear(self.d_model, 3 * self.n_heads * self.p_model)  # TODO 规范化到能够被切割整除
        self.Linear_Recover = nn.Linear(self.n_heads * self.p_model, d_model)  # TODO 还原到原本的维度

        self.dropout = nn.Dropout(dropout)

        # 写下对L的处理
        self.seq_model = seq_model
        self.SeqLinear = nn.Linear(seq_len, self.seq_model)
        self.SeqLinear_Recover = nn.Linear(self.seq_model, seq_len)

    def forward(self, input):
        # TODO input:[B, L, d_model]
        res = input.clone()  # TODO 方便后续进行残差连接
        bs, l, nh, pm = input.size(0), input.size(1), self.n_heads, self.p_model
        qkv = self.Linear(input)  # [B, L, 3 * n_heads * p_model]
        qkv = self.SeqLinear(qkv.permute(0, 2, 1)).permute(0, 2, 1)  # [B, S, D]
        q, k, v = qkv.chunk(3, dim=-1)  # Each one: [B, S, n_heads*p_model]=[B, S, D]
        q, k, v = map(lambda x: x.reshape(bs, -1, nh, pm), [q, k, v])  # [B, S, nh, pm]

        scores = torch.einsum('bnhd,bmhd->bhnm', (q, k))  # [B, nh, S, pm]

        scores_weight = F.softmax(scores / (self.p_model ** 0.5), dim=-1)  # [B, nh, S, pm]
        scores_weight = self.dropout(scores_weight)  # [B, nh, S, pm]

        attention = torch.einsum('bhnm,bmhd->bnhd', (scores_weight, v))  # [B, nh, S, pm]
        attention = attention.reshape(bs, -1, nh * pm)  # [B, S, nh*pm]
        attention = self.Linear_Recover(attention)  # [B, S, D]
        attention = self.SeqLinear_Recover(attention.permute(0, 2, 1)).permute(0, 2, 1)  # [B, L, D]

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
        x = self.Conv(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, L, N]

        return x

class Necker(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, seq_len, seq_model, o_model, dropout=0.1):
        super(Necker, self).__init__()
        self.BottleNecks = nn.ModuleList(
            [nn.Sequential(
                # SparseSelfAttention(d_model, n_heads, seq_len, seq_model, dropout),
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
                # SparseSelfAttention(d_model, n_heads, seq_len, seq_model, dropout),
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