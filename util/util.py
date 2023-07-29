import torch
import numpy as np
import PIL.Image as Image
import torchvision.transforms as transform
import cv2
import os
import torch.nn as nn


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
    unloader = transform.ToPILImage()

    input = img
    input_name = dir + '/' + str(i) + "_" + name + ".jpg"
    input = input.cpu().detach().numpy()
    input = np.clip(input * 255.0, 0, 255).astype(np.uint8)
    img = unloader(input.transpose([1, 2, 0]))
    img.save(input_name)


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


