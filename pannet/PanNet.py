import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as int


# -------------Initialization----------------------------------------
def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                variance_scaling_initializer(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


# -------------ResNet Block (One)----------------------------------------
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
    

# -----------------------------------------------------
class PanNet(nn.Module):
    def __init__(self):
        super(PanNet, self).__init__()

        channel = 32
        spectral_num = 8
        num_resblock = 4

        # self.deconv = nn.ConvTranspose2d(in_channels=spectral_num, out_channels=spectral_num, kernel_size=8, stride=4, padding=2, bias=True)
        self.upsample = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=True)
        self.conv0 = nn.Conv2d(in_channels=spectral_num + 1, out_channels=channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=spectral_num, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.layers = nn.ModuleList([])
        for _ in range(num_resblock):
            self.layers.append(Resblock())
        
        init_weights(self.layers, self.conv0, self.conv1)

    def forward(self, x, y, z):
        output_deconv = self.upsample(x)
        input = torch.cat([output_deconv, y], 1)
        rs = self.relu(self.conv0(input))
        for layer in self.layers:
            rs = layer(rs)
        output = self.conv1(rs)
        return output + z


# ----------------- End-Main-Part ------------------------------------
def variance_scaling_initializer(tensor):
    from scipy.stats import truncnorm

    def truncated_normal_(tensor, mean=0, std=1):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def variance_scaling(x, scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None):
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(x)
        if mode == "fan_in":
            scale /= max(1., fan_in)
        elif mode == "fan_out":
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)
        if distribution == "normal" or distribution == "truncated_normal":
            stddev = math.sqrt(scale) / .87962566103423978
        truncated_normal_(x, 0.0, stddev)
        return x/10*1.28

    variance_scaling(tensor)

    return tensor


# ----------------- summary ------------------------------------
def summaries(model, grad=False):
    if grad:
        from torchsummary import summary
        summary(model, input_size=[(8, 16, 16), (1, 64, 64), (8, 64, 64)], batch_size=1)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)


# model = PanNet().cuda()
# a = model.state_dict()
# summaries(model, grad=True)