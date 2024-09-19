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

        self.res00 = ResBlock(hidden_dim)
        self.res01 = ResBlock(hidden_dim)
        self.res02 = ResBlock(hidden_dim)
        self.res03 = ResBlock(hidden_dim)
        self.res10 = ResBlock(hidden_dim)
        self.res11 = ResBlock(hidden_dim)
        self.res12 = ResBlock(hidden_dim)
        self.res13 = ResBlock(hidden_dim)
        self.res20 = ResBlock(hidden_dim)
        self.res21 = ResBlock(hidden_dim)
        self.res22 = ResBlock(hidden_dim)
        self.res23 = ResBlock(hidden_dim)
        self.res30 = ResBlock(hidden_dim)
        self.res31 = ResBlock(hidden_dim)
        self.res32 = ResBlock(hidden_dim)
        self.res33 = ResBlock(hidden_dim)

        self.to_hrms0 = nn.Sequential(
            nn.Conv2d(hidden_dim, ms_dim, 3, 1, 1)
        )
        self.to_hrms1 = nn.Sequential(
            nn.Conv2d(hidden_dim, ms_dim, 3, 1, 1)
        )
        self.to_hrms2 = nn.Sequential(
            nn.Conv2d(hidden_dim, ms_dim, 3, 1, 1)
        )
        self.to_hrms = nn.Sequential(
            nn.Conv2d(hidden_dim, ms_dim, 3, 1, 1)
        )

    def forward(self, ms, pan):
        ms = self.upsample(ms)
        input = torch.cat([ms, pan], 1)
        x = self.raise_dim(input)
        skip0 = x

        x = self.res00(x)
        x = self.res01(x)
        x = self.res02(x)
        x = self.res03(x)
        x = x + skip0
        skip1 = x
        output0 = self.to_hrms0(x)

        x = self.res10(x)
        x = self.res11(x)
        x = self.res12(x)
        x = self.res13(x)
        x = x + skip1
        skip2 = x
        output1 = self.to_hrms1(x)

        x = self.res20(x)
        x = self.res21(x)
        x = self.res22(x)
        x = self.res23(x)
        x = x + skip2
        skip3 = x
        output2 = self.to_hrms2(x)

        x = self.res30(x)
        x = self.res31(x)
        x = self.res32(x)
        x = self.res33(x)
        x = x + skip3
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







