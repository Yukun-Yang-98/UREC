import torch
import torch.nn as nn
import torch.nn.functional as F

def Guided_filter(input, guide_map, sigma=0.01):
    
    mean_gm = torch.mean(guide_map, dim=1, keepdim=True)
    mean_input = torch.mean(input, dim=1, keepdim=True)
    corr_gm = torch.mean(mean_gm*mean_gm, dim=1, keepdim=True)
    corr_input = torch.mean(mean_input*mean_input, dim=1, keepdim=True)

    var_gm = corr_gm - mean_gm*mean_gm
    cov_input_gm = corr_input - mean_gm*mean_input

    a = cov_input_gm / (var_gm + sigma)
    b = mean_input - a * mean_gm

    mean_a = torch.mean(a, dim=1, keepdim=True)
    mean_b = torch.mean(b, dim=1, keepdim=True)

    # output = mean_a*guide_map + mean_b
    output = mean_a*mean_gm + mean_b
    return output




def _diff_x(src, r):
    cum_src = src.cumsum(-2)

    left = cum_src[..., r:2*r + 1, :]
    middle = cum_src[..., 2*r + 1:, :] - cum_src[..., :-2*r - 1, :]
    right = cum_src[..., -1:, :] - cum_src[..., -2*r - 1:-r - 1, :]

    output = torch.cat([left, middle, right], -2)

    return output

def _diff_y(src, r):
    cum_src = src.cumsum(-1)

    left = cum_src[..., r:2*r + 1]
    middle = cum_src[..., 2*r + 1:] - cum_src[..., :-2*r - 1]
    right = cum_src[..., -1:] - cum_src[..., -2*r - 1:-r - 1]

    output = torch.cat([left, middle, right], -1)

    return output

def boxfilter2d(src, radius):
    return _diff_y(_diff_x(src, radius), radius)

def guidedfilter2d_gray(guide, src, radius=60, eps=0.0001, scale=None):
    """guided filter for a gray scale guide image
    
    Parameters
    -----
    guide: (B, 1, H, W)-dim torch.Tensor
        guide image
    src: (B, C, H, W)-dim torch.Tensor
        filtering image
    radius: int
        filter radius
    eps: float
        regularization coefficient
    """
    if guide.ndim == 3:
        guide = guide[:, None]
    if src.ndim == 3:
        src = src[:, None]

    if scale is not None:
        guide_sub = guide.clone()
        src = F.interpolate(src, scale_factor=1./scale, mode="nearest")
        guide = F.interpolate(guide, scale_factor=1./scale, mode="nearest")
        radius = radius // scale

    ones = torch.ones_like(guide)
    N = boxfilter2d(ones, radius)

    mean_I = boxfilter2d(guide, radius) / N
    mean_p = boxfilter2d(src, radius) / N
    mean_Ip = boxfilter2d(guide*src, radius) / N
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = boxfilter2d(guide*guide, radius) / N
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = boxfilter2d(a, radius) / N
    mean_b = boxfilter2d(b, radius) / N

    if scale is not None:
        guide = guide_sub
        mean_a = F.interpolate(mean_a, guide.shape[-2:], mode='bilinear')
        mean_b = F.interpolate(mean_b, guide.shape[-2:], mode='bilinear')

    q = mean_a * guide + mean_b
    return q