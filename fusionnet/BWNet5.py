import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as int
from einops import rearrange, repeat


class Resblock(nn.Module):
    def __init__(self):
        super(Resblock, self).__init__()

        channel = 32
        self.conv20 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv21 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        rs1 = self.relu(self.conv20(x))
        rs1 = self.conv21(rs1)
        rs = torch.add(x, rs1)
        return rs


class Attention(nn.Module):
    def __init__(self, dim, num_res2blocks, dropout=0.):
        super().__init__()

        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, num_res2blocks),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _ = x.shape
        q = self.to_q(x)
        k = self.to_k(x)
        # attn = (q.transpose(-2, -1) @ k) * self.temperature / (n / (64 * 64))
        # attn = (q.transpose(-2, -1) @ k) * self.temperature / (n / 4)
        attn = (q.transpose(-2, -1) @ k) / (n / 4)
        out = self.to_out(attn).softmax(dim=-1)
        return out.transpose(-2, -1)


class BandSelectBlock(nn.Module):
    def __init__(self, dim, num_res2blocks):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_res2blocks)
        self.temperature = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, input, x):
        H = input.shape[2]
        input = self.norm(rearrange(input, 'B C H W -> B (H W) C', H=H))
        # input = rearrange(input, 'B C H W -> B (H W) C', H=H)
        attn = self.attn(input)  # B x num_res2blocks x C
        x = torch.stack(x, dim=0).transpose(0, 1)  # B x num_res2blocks x C x H x W
        output = torch.sum(attn.unsqueeze(-1).unsqueeze(-1) * x, dim=1) # + self.temperature  # B x C x H x W
        return output


class BWNet_PanNet(nn.Module):
    def __init__(self):
        super(BWNet_PanNet, self).__init__()

        channel = 32
        spectral_num = 8
        num_resblock = 4

        # self.deconv = nn.ConvTranspose2d(in_channels=spectral_num, out_channels=spectral_num, kernel_size=8, stride=4,
        # padding=2, bias=True)
        self.upsample = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=True)
        self.conv0 = nn.Conv2d(in_channels=spectral_num + 1, out_channels=channel, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=spectral_num, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.to_output = BandSelectBlock(channel, num_resblock)
        self.relu = nn.ReLU(inplace=True)
        self.layers = nn.ModuleList([])
        for _ in range(num_resblock):
            self.layers.append(Resblock())

    def forward(self, x, y, z):
        output_upsample = self.upsample(x)
        input = torch.cat([output_upsample, y], 1)
        rs = self.relu(self.conv0(input))
        band_select = rs
        
        output_feature_map_list = []
        for layer in self.layers:
            rs = layer(rs)
            output_feature_map_list.append(rs)

        output = self.to_output(band_select, output_feature_map_list)
        output = self.conv1(output)

        return output + z
    
    
class BWNet_FusionNet(nn.Module):
    def __init__(self):
        super(BWNet_FusionNet, self).__init__()

        channel = 32
        spectral_num = 8
        num_resblock = 4
        self.spectral_num = spectral_num

        self.conv0 = nn.Conv2d(in_channels=spectral_num, out_channels=channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=spectral_num, kernel_size=3, stride=1, padding=1, bias=True)
        self.to_output = BandSelectBlock(channel, num_resblock)
        self.relu = nn.ReLU(inplace=True)
        self.layers = nn.ModuleList([])
        for _ in range(num_resblock):
            self.layers.append(Resblock())

    def forward(self, x, y):
        y = y.repeat(1, self.spectral_num, 1, 1)
        input = y - x
        rs = self.relu(self.conv0(input))
        band_select = rs
        
        output_feature_map_list = []
        for layer in self.layers:
            rs = layer(rs)
            output_feature_map_list.append(rs)

        output = self.to_output(band_select, output_feature_map_list)
        output = self.conv1(output)

        return output + x


def summaries(model, grad=False):
    if grad:
        from torchsummary import summary
        summary(model, input_size=[(8, 64, 64), (1, 64, 64)], batch_size=1)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)
