import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from einops import rearrange, repeat

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

class LAConv2D(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, use_bias=True):
        super(LAConv2D, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = use_bias
        
        # Generating local adaptive weights
        self.attention1=nn.Sequential(
            nn.Conv2d(in_planes, kernel_size**2, kernel_size, stride, padding),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(kernel_size**2,kernel_size**2,1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(kernel_size ** 2, kernel_size ** 2, 1),
            nn.Sigmoid()
        )
        if use_bias: # Global local adaptive weights
            self.attention3=nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_planes,out_planes,1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_planes,out_planes,1)
            )

        conv1=nn.Conv2d(in_planes,out_planes,kernel_size,stride,padding,dilation,groups)
        self.weight=conv1.weight # m, n, k, k


    def forward(self,x):
        (b, n, H, W) = x.shape
        m=self.out_planes
        k=self.kernel_size
        n_H = 1 + int((H + 2 * self.padding - k) / self.stride)
        n_W = 1 + int((W + 2 * self.padding - k) / self.stride)
        atw1=self.attention1(x) #b,k*k,n_H,n_W
        #atw2=self.attention2(x) #b,n*k*k,n_H,n_W

        atw1=atw1.permute([0,2,3,1]) #b,n_H,n_W,k*k
        atw1=atw1.unsqueeze(3).repeat([1,1,1,n,1]) #b,n_H,n_W,n,k*k
        atw1=atw1.view(b,n_H,n_W,n*k*k) #b,n_H,n_W,n*k*k

        #atw2=atw2.permute([0,2,3,1]) #b,n_H,n_W,n*k*k

        atw=atw1#*atw2 #b,n_H,n_W,n*k*k
        atw=atw.view(b,n_H*n_W,n*k*k) #b,n_H*n_W,n*k*k
        atw=atw.permute([0,2,1]) #b,n*k*k,n_H*n_W

        kx=F.unfold(x,kernel_size=k,stride=self.stride,padding=self.padding) #b,n*k*k,n_H*n_W
        atx=atw*kx #b,n*k*k,n_H*n_W

        atx=atx.permute([0,2,1]) #b,n_H*n_W,n*k*k
        atx=atx.view(1,b*n_H*n_W,n*k*k) #1,b*n_H*n_W,n*k*k

        w=self.weight.view(m,n*k*k) #m,n*k*k
        w=w.permute([1,0]) #n*k*k,m
        y=torch.matmul(atx,w) #1,b*n_H*n_W,m
        y=y.view(b,n_H*n_W,m) #b,n_H*n_W,m
        if self.bias==True:
            bias=self.attention3(x) #b,m,1,1
            bias=bias.view(b,m).unsqueeze(1) #b,1,m
            bias=bias.repeat([1,n_H*n_W,1]) #b,n_H*n_W,m
            y=y+bias #b,n_H*n_W,m

        y=y.permute([0,2,1]) #b,m,n_H*n_W
        y=F.fold(y,output_size=(n_H,n_W),kernel_size=1) #b,m,n_H,n_W
        return y


# LAC_ResBlocks
class LACRB(nn.Module):
    def __init__(self, in_planes, ms_dim):
        super(LACRB, self).__init__()
        self.conv1=LAConv2D(in_planes,in_planes,3,1,1,use_bias=True)
        self.relu1=nn.LeakyReLU(inplace=True)
        self.conv2=LAConv2D(in_planes,in_planes,3,1,1,use_bias=True)

    def forward(self,x):
        res=self.conv1(x)
        res=self.relu1(res)
        res=self.conv2(res)
        x=x+res
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_lacrbs, dropout=0.):
        super().__init__()

        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.hidden_dim = dim * 2
        self.linear_transform = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, self.hidden_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, num_lacrbs),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _ = x.shape
        # q = self.to_q(x)
        # k = self.to_k(x)
        cov = x.transpose(-2, -1) @ x
        attn = self.linear_transform(cov) * self.temperature / (n / (64 * 64))
        out = self.to_out(attn).softmax(dim=-1)
        return out.transpose(-2, -1)


class BandSelectBlock(nn.Module):
    def __init__(self, dim, num_lacrbs):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_lacrbs)
        self.temperature = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, input, x):
        H = input.shape[2]
        input = self.norm(rearrange(input, 'B C H W -> B (H W) C', H=H))
        attn = self.attn(input)  # B x num_res2blocks x C
        x = torch.stack(x, dim=0).transpose(0, 1)  # B x num_res2blocks x C x H x W
        output = torch.sum(attn.unsqueeze(-1).unsqueeze(-1) * x, dim=1) + self.temperature  # B x C x H x W
        return output


class BWNet(nn.Module):
    def __init__(self, pan_dim=1, ms_dim=8, channel=32, num_lacrbs=5):
        super().__init__()

        self.ms_dim = ms_dim
        self.raise_dim = nn.Sequential(
            LAConv2D(pan_dim + ms_dim, channel, 3, 1, 1, use_bias=True),
            nn.LeakyReLU(inplace=True)
        )
        self.layers = nn.ModuleList([])
        for _ in range(num_lacrbs):
            self.layers.append(LACRB(channel, ms_dim))
        self.bw_output = BandSelectBlock(channel, num_lacrbs)
        self.to_output = nn.Sequential(
            LAConv2D(channel, ms_dim, 3, 1, 1, use_bias=True)
        )

    def forward(self, ms, pan):
        ms = F.interpolate(ms, scale_factor=4, mode='bicubic')
        input = torch.concatenate([pan, ms], dim=1)
        x = self.raise_dim(input)
        raise_output = x
        feature_list = []

        for layer in self.layers:
            x = layer(x)
            feature_list.append(x)
        output = self.bw_output(raise_output, feature_list)

        return self.to_output(output) + ms


def summaries(model, grad=False):
    if grad:
        from torchsummary import summary
        summary(model, input_size=[(8, 16, 16), (1, 64, 64)], batch_size=1)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

model = BWNet().cuda()
summaries(model, grad=True)






