import torch
import numpy as np
import PIL.Image as Image
import torchvision.transforms as transform
import cv2
import os
import torch.nn as nn
import torchvision.transforms.functional as F


def tensor_gray(input):
    batch = input.shape[0]
    newinput = input.clone()
    for i in range(0, batch):
        newinput[i, 0, :, :] = input[i, 0, :, :] / 3
        newinput[i, 0, :, :] = newinput[i, 0, :, :] + input[i, 1, :, :] / 3
        newinput[i, 0, :, :] = newinput[i, 0, :, :] + input[i, 2, :, :] / 3
    return newinput[:, 0, :, :]


def tensor_gray_3(input):
    newinput = input.clone()
    newinput[0, :, :] = input[0, :, :] / 3
    newinput[0, :, :] = newinput[0, :, :] + input[1, :, :] / 3
    newinput[0, :, :] = newinput[0, :, :] + input[2, :, :] / 3
    return newinput[0, :, :]


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def RGB2BGR(input):
    output = input
    output[0, :, :] = input[2, :, :]
    output[2, :, :] = input[0, :, :]
    return output


def MAXC(input):
    R = input[:, 0:1, :, :]
    G = input[:, 1:2, :, :]
    B = input[:, 2:3, :, :]
    out = torch.max(R, torch.max(G, B))
    # out = out.unsqueeze(1)
    return out


def sample(R, L, i, img, name):
    unloader = transform.ToPILImage()

    input = img
    input_name = name + '/' + str(i) + "_pre.jpg"
    input = input.cpu().detach().numpy()
    input = np.clip(input * 255.0, 0, 255).astype(np.uint8)
    # input = RGB2BGR(input)
    # print(input.shape)
    # img = Image.fromarray(np.uint8(input))
    img = unloader(input.transpose([1, 2, 0]))
    img.save(input_name)

    L = torch.cat([L, L, L])
    out_no_noise = R * L
    out_no_noise_name = name + '/' + str(i) + "_gen.jpg"
    out_no_noise = out_no_noise.cpu().detach().numpy()
    out_no_noise = np.clip(out_no_noise * 255.0, 0, 255).astype(np.uint8)
    # img_no_noise = Image.fromarray(np.uint8(out_no_noise))
    img_no_noise = unloader(out_no_noise.transpose([1, 2, 0]))
    img_no_noise.save(out_no_noise_name)

    out_R_name = name + '/' + str(i) + "_R.jpg"
    out_R = R.cpu().detach().numpy()
    out_R = np.clip(out_R * 255.0, 0, 255).astype(np.uint8)
    img_R = unloader(out_R.transpose([1, 2, 0]))
    img_R.save(out_R_name)


def sample_single_img(i, img, name, dir):
    input_name = dir + '/' + str(i) + "_" + name + ".png"
    # unloader = transform.ToPILImage()
    # input = img
    # input = input.cpu().detach().numpy()
    # input = np.clip(input * 255.0, 0, 255).astype(np.uint8)
    # img = unloader(input.transpose([1, 2, 0]))
    # img.save(input_name)
    if img.dtype != torch.uint8:
        img = (img * 255).clamp(0, 255).byte()
    img_pil = F.to_pil_image(img)
    img_pil.save(input_name, format='png', compress_level=0)


def sample_gray_img(i, img_gray, name, dir):
    if len(img_gray.shape) == 3:
        img_gray = img_gray.squeeze(0)
    input = img_gray
    input_name = dir + '/' + str(i) + "_" + name + ".jpg"
    input = input.cpu().detach().numpy()
    input = np.clip(input * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(input)
    img.convert('L').save(input_name)


def get_dir_name(path, name):
    dirs = os.listdir(path)
    for i in range(1, 1000):
        newname = name + str(i)
        if newname not in dirs:
            name = newname
            break
    print('run dir is:   ', path + '/' + name)
    return path + '/' + name



def cal_HSL(input):
    ## calculate HSL from RGB
    R = input[:, 0:1, :, :]
    G = input[:, 1:2, :, :]
    B = input[:, 2:3, :, :]
    Max = torch.max(R, torch.max(G, B))
    Min = torch.min(R, torch.min(G, B))
    Minus = Max - Min + 0.001
    L = (Max + Min) / 2
    zeros = torch.zeros_like(L)
    S = torch.where(L > 0.5, Minus / (2.001 - 2 * L), Minus / (2 * L + 0.001))
    S = torch.where(L == 0, zeros, S)
    # S = torch.where(Max == Min, zeros, S)
    H = torch.where(Max == R, (G - B) / Minus / 6, zeros)
    H = torch.where(G < B, 1 + (G - B) / Minus / 6, H)
    H = torch.where(Max == G, 1 / 3 + (B - R) / Minus / 6, H)
    H = torch.where(Max == B, 2 / 3 + (R - G) / Minus / 6, H)
    H = torch.where(Max == Min, zeros, H)
    H = H - torch.floor(H)
    return H, S, L


class RGB_HSV(nn.Module):
    def __init__(self, eps=1e-8):
        super(RGB_HSV, self).__init__()
        self.eps = eps

    def rgb_to_hsv(self, img):
        hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)
        hue[ img[:,2]==img.max(1)[0] ] = 4.0 + ( (img[:,0]-img[:,1]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,2]==img.max(1)[0] ]
        hue[ img[:,1]==img.max(1)[0] ] = 2.0 + ( (img[:,2]-img[:,0]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,1]==img.max(1)[0] ]
        hue[ img[:,0]==img.max(1)[0] ] = (0.0 + ( (img[:,1]-img[:,2]) / ( img.max(1)[0] - img.min(1)[0] + self.eps) ) [ img[:,0]==img.max(1)[0] ]) % 6
        hue[img.min(1)[0]==img.max(1)[0]] = 0.0
        hue = hue/6
        saturation = ( img.max(1)[0] - img.min(1)[0] ) / ( img.max(1)[0] + self.eps )
        saturation[ img.max(1)[0]==0 ] = 0
        value = img.max(1)[0]        
        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)
        hsv = torch.cat([hue, saturation, value],dim=1)
        # return hsv
        return hue

    def hsv_to_rgb(self, hsv):
        h,s,v = hsv[:,0,:,:],hsv[:,1,:,:],hsv[:,2,:,:]
        #对出界值的处理
        h = h%1
        s = torch.clamp(s,0,1)
        v = torch.clamp(v,0,1)
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)        
        hi = torch.floor(h * 6)
        f = h * 6 - hi
        p = v * (1 - s)
        q = v * (1 - (f * s))
        t = v * (1 - ((1 - f) * s))        
        hi0 = hi==0
        hi1 = hi==1
        hi2 = hi==2
        hi3 = hi==3
        hi4 = hi==4
        hi5 = hi==5        
        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]        
        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]        
        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]        
        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]
        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]        
        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]        
        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        return rgb

