import torch
import torch.nn as nn


def get_conv2d_layer(in_c, out_c, k, s, p=0, dilation=1, groups=1):
    return nn.Conv2d(in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=k,
                    stride=s,
                    padding=p,
                    padding_mode='replicate',
                    dilation=dilation, groups=groups)

    
class Decom_U3_L_sm(nn.Module):
    def __init__(self):
        super().__init__()
        self.decom = nn.Sequential(
            get_conv2d_layer(in_c=3, out_c=16, k=3, s=1, p=1),
            nn.ReLU(inplace=True),
            get_conv2d_layer(in_c=16, out_c=16, k=3, s=1, p=1),
            nn.ReLU(inplace=True),
            get_conv2d_layer(in_c=16, out_c=3, k=3, s=1, p=1),
            nn.ReLU()
        )
    def forward(self, input):
        output = self.decom(input)
        L = output
        R = input / (L+0.000001)
        return R, L
    
    
class Decom_1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.decom = nn.Sequential(
            get_conv2d_layer(in_c=1, out_c=16, k=3, s=1, p=1),
            nn.ReLU(inplace=True),
            get_conv2d_layer(in_c=16, out_c=16, k=3, s=1, p=1),
            nn.ReLU(inplace=True),
            get_conv2d_layer(in_c=16, out_c=1, k=3, s=1, p=1),
            nn.ReLU()
        )
    def forward(self, input):
        input_mean = torch.mean(input, dim=1, keepdim=True)
        output = self.decom(input_mean)
        L = output
        R = input / (L+0.000001)
        return R, L
 