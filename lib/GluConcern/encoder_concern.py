import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class ANNBlock(nn.Module):
    def __init__(self, d_model, h_model, activation, dropout):
        super(ANNBlock, self).__init__()
        self.Linear_Projection = nn.Linear(d_model, h_model)
        self.activation = activation_selection(activation)
        self.dropout = nn.Dropout(dropout)
        self.Linear_Recover = nn.Linear(h_model, d_model)

    def forward(self, input_data):
        # input_data [B, N, L]
        input_data = self.Linear_Projection(input_data)  # [B, N, H]
        input_data = self.activation(input_data)  # [B, N, H]
        input_data = self.dropout(input_data)  # [B, N, H]
        input_data = self.Linear_Recover(input_data)  # [B, N, L]

        return input_data  # [B, N, L]


class Processor(nn.Module):
    def __init__(self, control_ratio, d_model, hd_model, c_model, cd_model, activation, dropout):
        super(Processor, self).__init__()
        # 采用两个ANN对数据对后两个维度进行处理
        self.control_ratio = control_ratio
        self.ANN_L = ANNBlock(d_model, hd_model, activation, dropout)
        self.ANN_C = ANNBlock(c_model, cd_model, activation, dropout)


    def forward(self, input_data):
        # input_data [B, N, L]
        # 为通道维度处理做准备
        # input_data_C = (input_data.clone()).permute(0, 2, 1)  # [B, L, N]

        input_data_L = self.ANN_L(input_data)  # [B, N, L]
        input_data_C = self.ANN_C(input_data_L.permute(0, 2, 1)).permute(0, 2, 1)  # [B, L, N]

        # 实现对比例的自动调节
        input_data = input_data_C

        return input_data  # [B, N, L]




class EncoderProcessor(nn.Module):
    def __init__(self, d_model:int, in_channel:int, out_channel_ratio:int, control_ratio, hd_model:int, cd_model:int, activation:str, dropout):
        super(EncoderProcessor, self).__init__()
        # self.LayerNorm = nn.LayerNorm(d_model)
        self.ConvExtender = nn.Conv1d(in_channels=in_channel, out_channels=int(in_channel*out_channel_ratio), kernel_size=3, padding=1)
        self.ConvRestorer = nn.Conv1d(in_channels=int(in_channel*out_channel_ratio), out_channels=int(in_channel*out_channel_ratio), kernel_size=1)
        self.ConvRecover = nn.Conv1d(in_channels=int(in_channel*out_channel_ratio), out_channels=int(in_channel*out_channel_ratio), kernel_size=3, padding=1)

        self.activation = activation_selection(activation)

        self.processor = Processor(control_ratio, d_model, hd_model, in_channel*out_channel_ratio, cd_model, activation, dropout)

        self.ConvEnder = nn.Conv1d(in_channels=int(in_channel*out_channel_ratio), out_channels=in_channel, kernel_size=1)

    def forward(self, g_data):
        # g_data [B, L, N]
        # 为残差链接做准备
        g_data_res = g_data.clone()  # [B, L, N]

        # 主要的处理
        # g_data_norm = self.LayerNorm(g_data.permute(0,2,1))  # [B, N, L]
        g_data_norm = g_data.permute(0, 2, 1)  # [B, N, L]
        g_data_1 = self.ConvExtender(g_data_norm)  # [B, KN, L]
        g_data_2 = self.ConvRestorer(g_data_1)  # [B, KN, L]
        g_data_3 = self.ConvRecover(g_data_2)  # [B, KN, L]

        g_data_activation = self.activation(g_data_3)  # [B, KN, L]

        # N和L维度都进行混合  [B, KN, L]
        g_data_processed = self.processor(g_data_activation)  # [B, KN, L]

        g_data_end = self.ConvEnder(g_data_processed)  # [B, N, L]
        g_state_end = g_data_end.permute(0,2,1)  # [B, L, N]

        return g_state_end + g_data_res  # [B, L, N]


class EncoderConcern(nn.Module):
    def __init__(self, d_model:int, in_channel:int, out_channel_ratio:int, control_ratio, hd_model:int, cd_model:int, activation:str, dropout):
        super(EncoderConcern, self).__init__()
        self.EncoderProcessor = EncoderProcessor(d_model, in_channel, out_channel_ratio, control_ratio, hd_model, cd_model, activation, dropout)
        # self.LayerNorm = nn.LayerNorm(d_model)
        # self.FFT = torch.fft.fft
        #
        # # 要分为两条支路分别处理幅频和相频
        # # 相频一般不处理，下面仅写幅频处理，仍然用ANN网络
        # self.processor = Processor(control_ratio, d_model, hd_model, in_channel, cd_model, activation, dropout)
        # self.ConvExtender = nn.Conv1d(in_channels=in_channel, out_channels=int(in_channel * out_channel_ratio), kernel_size=3, padding=1)
        # self.ConvRestorer = nn.Conv1d(in_channels=int(in_channel * out_channel_ratio), out_channels=int(in_channel * out_channel_ratio), kernel_size=1)
        # self.activation = activation_selection(activation)
        # self.ConvEnder = nn.Conv1d(in_channels=int(in_channel*out_channel_ratio), out_channels=in_channel, kernel_size=1)

    def forward(self, g_data):
        # g_data [B, L, N]
        g_data = self.EncoderProcessor(g_data)  # [B, L, N]

        # # 为借鉴ECMB的思想准备一个残差
        # g_data_res= g_data.clone()
        #
        # # 完成幅频分支
        # # g_data = self.LayerNorm(g_data.permute(0,2,1))  # [B, N, L]
        # g_data = g_data.permute(0, 2, 1)  # [B, N, L]
        # g_data_fft = self.FFT(g_data)  # 得到的仍然是[B, N, L] 需要torch.abs得到幅频，torch.angle得到相频
        # g_data_mag = torch.abs(g_data_fft)  # [B, N, L]
        # g_data_pha = torch.angle(g_data_fft)  # [B, N, L]
        #
        # g_data_mag = self.processor(g_data_mag)  # [B, N, L]
        #
        # real = g_data_mag * torch.cos(g_data_pha)
        # imag = g_data_mag * torch.sin(g_data_pha)
        #
        # # 合成复数频域张量
        # g_data_fft_reconstructed = torch.complex(real, imag)
        #
        # # 然后做 IFFT
        # g_data_ifft = torch.fft.ifft(g_data_fft_reconstructed, dim=2).real  # 恢复时域实数信号 [B, N, L]
        # g_data_ifft = g_data_ifft.permute(0, 2, 1)  # [B, L, N]
        #
        # g_data = g_data_ifft  # 这里是点乘，一一对应乘，结果还是[B, L, N]
        # res = g_data_res + g_data  # [B, L, N]

        return g_data  # [B, L, N]