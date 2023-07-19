import torch.nn as nn
import torch
from torch.nn.modules.linear import Identity
import math
import torch.nn.functional as F


def get_conv2d_layer(in_c, out_c, k, s, p=0, dilation=1, groups=1):
    return nn.Conv2d(in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=k,
                    stride=s,
                    padding=p,dilation=dilation, groups=groups,
                    padding_mode='reflect')


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Decom_U3_ReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.decom = nn.Sequential(
            get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1),
            nn.ReLU(inplace=True),
            get_conv2d_layer(in_c=32, out_c=32, k=3, s=1, p=1),
            nn.ReLU(inplace=True),
            get_conv2d_layer(in_c=32, out_c=32, k=3, s=1, p=1),
            nn.ReLU(inplace=True),
            get_conv2d_layer(in_c=32, out_c=6, k=3, s=1, p=1),
            nn.ReLU()
        )
    def forward(self, input):
        output = self.decom(input)
        R = output[:, 0:3, :, :]
        L = output[:, 3:6, :, :]
        return R, L
 
class Km0_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.convI = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.reluI = nn.ReLU(inplace=True)
        self.convR = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.reluR = nn.ReLU(inplace=True)
        self.convRm = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.reluRm = nn.ReLU(inplace=True)
        self.convR0out = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.reluR0out = nn.ReLU(inplace=True)
        self.convS = get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1)
        self.reluS = nn.ReLU(inplace=True)
        self.convg = get_conv2d_layer(in_c=1, out_c=32, k=3, s=1, p=1)
        self.relug = nn.ReLU(inplace=True)

        self.se_layer = SELayer(channel=160)
        self.conv4 = get_conv2d_layer(in_c=160, out_c=64, k=3, s=1, p=1)
        self.relu4 = nn.ReLU(inplace=True)

        # self.convmid1 = get_conv2d_layer(in_c=64, out_c=64, k=3, s=1, p=1)
        # self.relumid1 = nn.ReLU(inplace=True)
        # self.convmid2 = get_conv2d_layer(in_c=64, out_c=64, k=3, s=1, p=1)
        # self.relumid2 = nn.ReLU(inplace=True)
        # self.convmid3 = get_conv2d_layer(in_c=64, out_c=64, k=3, s=1, p=1)
        # self.relumid3 = nn.ReLU(inplace=True)

        self.conv5 = get_conv2d_layer(in_c=64, out_c=16, k=3, s=1, p=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = get_conv2d_layer(in_c=16, out_c=2, k=3, s=1, p=1)
        self.relu6 = nn.ReLU(inplace=True)

    def forward(self, I, R, Rm, S, guide_map):
        I_fs = self.reluI(self.convI(I))
        R_fs = self.reluR(self.convR(R))
        Rm_fs = self.reluRm(self.convRm(Rm))
        # R0out_fs = self.reluR0out(self.convR0out(R0out))
        S_fs = self.reluS(self.convS(S))
        gm_fs = self.relug(self.convg(guide_map))
        # inf = torch.cat([I_fs, R_fs, Rm_fs, R0out_fs, gm_fs], dim=1)
        inf = torch.cat([I_fs, R_fs, Rm_fs, S_fs, gm_fs], dim=1)
        se_inf = self.se_layer(inf)
        x1 = self.relu4(self.conv4(se_inf))

        # x1 = self.relumid1(self.convmid1(x1))
        # x1 = self.relumid2(self.convmid2(x1))
        # x1 = self.relumid3(self.convmid3(x1))

        x2 = self.relu5(self.conv5(x1))
        x3 = self.relu6(self.conv6(x2))
        return x3    