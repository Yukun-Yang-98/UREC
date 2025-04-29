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

     
    
class Km0_net_1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.convI = get_conv2d_layer(in_c=1, out_c=16, k=3, s=1, p=1)
        self.reluI = nn.ReLU(inplace=True)
        self.convR = get_conv2d_layer(in_c=1, out_c=16, k=3, s=1, p=1)
        self.reluR = nn.ReLU(inplace=True)
        self.convRm = get_conv2d_layer(in_c=1, out_c=16, k=3, s=1, p=1)
        self.reluRm = nn.ReLU(inplace=True)
        self.convg = get_conv2d_layer(in_c=1, out_c=16, k=3, s=1, p=1)
        self.relug = nn.ReLU(inplace=True)

        self.se_layer = SELayer(channel=64)
        self.conv4 = get_conv2d_layer(in_c=64, out_c=32, k=3, s=1, p=1)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = get_conv2d_layer(in_c=32, out_c=16, k=3, s=1, p=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = get_conv2d_layer(in_c=16, out_c=2, k=3, s=1, p=1)
        self.relu6 = nn.ReLU(inplace=True)

    def forward(self, I, R, Rm, guide_map):
        I_fs = self.reluI(self.convI(I))
        R_fs = self.reluR(self.convR(R))
        Rm_fs = self.reluRm(self.convRm(Rm))
        gm_fs = self.relug(self.convg(guide_map))
        inf = torch.cat([I_fs, R_fs, Rm_fs, gm_fs], dim=1)
        se_inf = self.se_layer(inf)
        x1 = self.relu4(self.conv4(se_inf))
        x2 = self.relu5(self.conv5(x1))
        x3 = self.relu6(self.conv6(x2))
        # x3 = self.conv6(x2)
        return x3
    

class Km0_net_3d(nn.Module):
    def __init__(self):
        super().__init__()
        self.convI = get_conv2d_layer(in_c=3, out_c=16, k=3, s=1, p=1)
        self.reluI = nn.ReLU(inplace=True)
        self.convR = get_conv2d_layer(in_c=3, out_c=16, k=3, s=1, p=1)
        self.reluR = nn.ReLU(inplace=True)
        self.convRm = get_conv2d_layer(in_c=3, out_c=16, k=3, s=1, p=1)
        self.reluRm = nn.ReLU(inplace=True)
        self.convg = get_conv2d_layer(in_c=1, out_c=16, k=3, s=1, p=1)
        self.relug = nn.ReLU(inplace=True)

        self.se_layer = SELayer(channel=64)
        self.conv4 = get_conv2d_layer(in_c=64, out_c=32, k=3, s=1, p=1)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = get_conv2d_layer(in_c=32, out_c=16, k=3, s=1, p=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = get_conv2d_layer(in_c=16, out_c=2, k=3, s=1, p=1)
        self.relu6 = nn.ReLU(inplace=True)

    def forward(self, I, R, Rm, guide_map):
        I_fs = self.reluI(self.convI(I))
        R_fs = self.reluR(self.convR(R))
        Rm_fs = self.reluRm(self.convRm(Rm))
        gm_fs = self.relug(self.convg(guide_map))
        inf = torch.cat([I_fs, R_fs, Rm_fs, gm_fs], dim=1)
        se_inf = self.se_layer(inf)
        x1 = self.relu4(self.conv4(se_inf))
        x2 = self.relu5(self.conv5(x1))
        x3 = self.relu6(self.conv6(x2))
        # x3 = self.conv6(x2)
        return x3
    

class Guidemap_auto_net_sm(nn.Module):
    def __init__(self, points = 1):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()
        '''  calculate points and parameters  '''
        self.points = points
        self.out_layers = points*3+1
        self.conv1 = get_conv2d_layer(in_c=1, out_c=8, k=3, s=1, p=1)
        self.conv2 = get_conv2d_layer(in_c=8, out_c=8, k=3, s=1, p=1)
        self.conv3 = get_conv2d_layer(in_c=8, out_c=self.out_layers, k=3, s=1, p=1)
        self.bn4 = nn.BatchNorm2d(self.out_layers)

    def forward(self, I):
        x1 = self.relu(self.conv1(I))
        x2 = self.relu(self.conv2(x1))
        x4 = self.sig(self.conv3(x2))
        Ps = x4[:, 0:int(self.points*2), :, :]
        As = x4[:, int(self.points*2):int(self.out_layers), :, :]

        return Ps, As
    
def curve_change(gmap, Xs, Ys, As):
    '''  just work on V  '''
    map_out = gmap.clone()
    Vout = torch.zeros_like(gmap)
    xs = torch.split(Xs, 1, dim=1)
    ys = torch.split(Ys, 1, dim=1)
    ks = torch.split(As, 1, dim=1)

    x1 = torch.zeros_like(xs[0])
    y1 = torch.zeros_like(ys[0])
    x1 = x1.unsqueeze(2).unsqueeze(3)
    y1 = y1.unsqueeze(2).unsqueeze(3)
    for i in range(len(ks)):
        k = ks[i]
        V_cp = gmap.clone()
        if i == len(ks)-1:            
            x2 = torch.ones_like(xs[0])
            y2 = torch.ones_like(ys[0])
        else:
            x2 = xs[i]
            y2 = ys[i]
        x2 = x2.unsqueeze(2).unsqueeze(3)
        y2 = y2.unsqueeze(2).unsqueeze(3)
        thr1 = x1
        thr2 = x2
        thr1_exp = thr1
        thr2_exp = thr2
        mask1 = gmap>=thr1_exp
        mask2 = gmap<=thr2_exp
        Vmid = (y2-y1) / (torch.exp(k*x2)-torch.exp(k*x1)) * (torch.exp(k*V_cp)-torch.exp(k*x1)) + y1
        Vmid = Vmid*mask1*mask2
        Vout += Vmid
        x1 = x2
        y1 = y2
    map_out = Vout
    return map_out