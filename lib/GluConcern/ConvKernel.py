import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

# TODO 写一个对Temporal操作的多层卷积
class TimeConv(nn.Module):
    def __init__(self, tc_layers, seq_len, seq_model):
        super(TimeConv, self).__init__()
        self.tc_layers = tc_layers
        self.Linear = nn.Linear(seq_len, seq_model)
        self.seq_model = seq_model
        # TODO 做一个Multi-Scale的网络？
        self.ConvKernels = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(self.seq_model // (2**i), self.seq_model // (2**(i+1)), kernel_size=3, padding=1, stride=1, groups=1),  # TODO每个通道单独卷积
                    nn.Conv1d(self.seq_model // (2**(i+1)), self.seq_model // (2**(i+1)), kernel_size=1, stride=1, padding=0, groups=1),
                    nn.Hardswish()
                ) for i in range(self.tc_layers)  # 整体信息进行卷积
            ]
        )

        self.Linears = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.seq_model // (2**i), self.seq_model // (2**(i+1)))
                ) for i in range(self.tc_layers)
            ]
        )
        self.Linear_Recover = nn.Linear(self.seq_model // (2**self.tc_layers), seq_len)


    def forward(self, x):
        # x [B, L, N]
        x = x.permute(0, 2, 1)  # [B, N, L]
        x = self.Linear(x)  # [B, N, LD]
        for i in range(self.tc_layers):
            res = self.Linears[i](x).permute(0, 2, 1)  # [B, LD, N]
            x = self.ConvKernels[i](x.permute(0, 2, 1))  + res  # 每次卷积前去浓缩之前的全局信息
            x = x.permute(0, 2, 1)  # [B, N, ?]
        x = self.Linear_Recover(x)  # [B, N, L] TODO 这里需要残差网络进行连接吗？
        x = x.permute(0, 2, 1)  # [B, L, N]
        return x


class ChannelConv(nn.Module):
    def __init__(self, cc_layers, d_model):
        super(ChannelConv, self).__init__()
        self.cc_layers = cc_layers
        self.d_model = d_model
        # 做一个膨胀卷积？
        self.ConvEncoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.d_model // (2**i), self.d_model // (2**(i+1)), kernel_size=3, stride=1, padding=1, groups=1),
                nn.Hardswish()
            ) for i in range(self.cc_layers)
        ])

        self.ConvEncoders2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.d_model // (2**i), self.d_model // (2**(i+1)), kernel_size=1, stride=1, padding=0, groups=1),
            ) for i in range(self.cc_layers)
        ])
        self.Linears = nn.ModuleList(
            [
                nn.Linear(self.d_model // (2**i), self.d_model // (2**(i+1))) for i in range(self.cc_layers)
            ]
        )
        # self.ConvNeck = nn.Sequential(
        #         nn.Conv1d(self.d_model // (2**self.cc_layers), self.d_model // (2**self.cc_layers), kernel_size=1, stride=1, padding=0, groups=1),
        #         nn.Hardswish()
        # )
        self.ConvDecoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.d_model // (2**(self.cc_layers-i)), self.d_model // (2**(self.cc_layers-i-1)), kernel_size=3, stride=1, padding=1, groups=1),
                nn.Hardswish()
            ) for i in range(self.cc_layers)
        ])


    def forward(self, x):
        # x [B, L, N]
        Res = x.clone()
        Skips = [Res]
        for i in range(self.cc_layers):
            res = self.Linears[i](x).permute(0, 2, 1)  # [B, N//i, L]
            x = self.ConvEncoders[i](x.permute(0, 2, 1)) + self.ConvEncoders2[i](x.permute(0, 2, 1)) + res  # [B, N//i, L]
            Skips.append(x)  # [B, N//i, L]
            x = x.permute(0, 2, 1)  # [B, L, N//i]
        # x = self.ConvNeck(x.permute(0, 2, 1)).permute(0, 2, 1) + x  # [B, L, N/(2^layers)]
        Skips.reverse()
        for i in range(self.cc_layers):
            x = self.ConvDecoders[i](x.permute(0, 2, 1)+ Skips[i])
            x = x.permute(0, 2, 1)
        return x  # [B, L, N]



class ConvAll(nn.Module):
    def __init__(self,e_layers, d_layers, seq_len, seq_model, d_model):
        super(ConvAll, self).__init__()
        self.TimeConv = TimeConv(e_layers, seq_len, seq_model)
        self.ChannelConv = ChannelConv(d_layers, d_model)

    def forward(self, x):
        Res = x.clone()
        x = self.TimeConv(x)
        channel_x = Res.clone()
        channel_x = self.ChannelConv(channel_x)
        x = channel_x + x  # [B, L, N]
        return x - Res