import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


class ResBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv0 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv1 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        rs1 = self.relu(self.conv0(x))
        rs1 = self.conv1(rs1)
        rs = torch.add(x, rs1)
        return rs


class BWNet(nn.Module):
    def __init__(self, pan_dim, ms_dim, hidden_dim):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Conv2d(ms_dim, ms_dim * 16, 3, 1, 1),
            nn.PixelShuffle(4)
        )

        self.raise_dim = nn.Sequential(
            nn.Conv2d(ms_dim + pan_dim, hidden_dim, 3, 1, 1),
            nn.LeakyReLU()
        )

        self.res1 = ResBlock(hidden_dim)
        self.res2 = ResBlock(hidden_dim)
        self.res3 = ResBlock(hidden_dim)
        self.res4 = ResBlock(hidden_dim)

        self.to_hrms0 = nn.Conv2d(hidden_dim, ms_dim, 3, 1, 1)
        self.to_hrms1 = nn.Conv2d(hidden_dim, ms_dim, 3, 1, 1)
        self.to_hrms2 = nn.Conv2d(hidden_dim, ms_dim, 3, 1, 1)
        self.to_hrms = nn.Conv2d(hidden_dim, ms_dim, 3, 1, 1)

    def forward(self, ms, pan):
        ms = self.upsample(ms)
        input = torch.cat([ms, pan], 1)
        x = self.raise_dim(input)
        skip0 = x

        x = self.res1(x)
        x = x + skip0
        skip1 = x
        output0 = self.to_hrms0(x)

        x = self.res2(x)
        x = x + skip0 + skip1
        skip2 = x
        output1 = self.to_hrms1(x)

        x = self.res3(x)
        x = x + skip0 + skip1 + skip2
        skip3 = x
        output2 = self.to_hrms2(x)

        x = self.res4(x)
        x = x + skip0 + skip1 + skip2 + skip3
        output3 = self.to_hrms(x)

        return output0 + ms, output1 + ms, output2 + ms, output3 + ms


def summaries(model, grad=False):
    if grad:
        from torchsummary import summary
        summary(model, input_size=[(8, 16, 16), (1, 64, 64)], batch_size=1)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)


# model = BWNet(1, 8, 32).cuda()
# summaries(model, grad=True)







