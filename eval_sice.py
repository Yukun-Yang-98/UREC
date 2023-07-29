import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import shutil
import cv2
import math
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
import lpips
from PIL import Image
from tqdm import tqdm
from util.util import *

def getparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_cpus", type=int, default=32)
    parser.add_argument("--data_path", type=str, default='./test/SICE_Stkm3_19/')   ### Here is the path of results on SICE dataset
    parser.add_argument("--img_size", type=int, default=[900, 1200])
    parser.add_argument("--tail", type=str, default='MYkm3' + '19')

    print(parser.parse_args())
    return parser.parse_args()

opt = getparser()
under_ssims = []
under_psnrs = []
under_lpips = []
over_lpips = []
over_ssims = []
over_psnrs = []

trans = transforms.Compose([
	transforms.ToTensor()
])

loss_fn = lpips.LPIPS(net='vgg')
loss_fn.cuda()

def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_lpips(img1, img2):
   '''
   img1, img2: tensor.cuda()
   '''
   loss_fn = lpips.LPIPS(net='vgg')
   loss_fn = loss_fn.cuda()
   dist = loss_fn.forward(img1, img2)
   return dist

def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def Test_train(std_imgs_path, test_dir):
    
    print("Start calculate parameters between ", std_imgs_path, "and", test_dir)
    images = os.listdir(test_dir)
    images.sort()
    stdimages = os.listdir(std_imgs_path)
    stdimages.sort()
    # print(len(images))
    # print(len(stdimages))

    ssims = []
    psnrs = []
    lpipses = []

    i = 0
    img_f3 = ''
    now_std = 0
    stdimg = None
    stdimg_path = ''
    for img in tqdm(images):
        img_path = test_dir + img
        img_3 = img[0:3]
        if not img_3 == img_f3:
            img_f3 = img_3
            stdimg_path = std_imgs_path + stdimages[now_std]
            stdimg = cv2.imread(stdimg_path, 0)
            stdimg = cv2.resize(stdimg, (opt.img_size[1], opt.img_size[0]), interpolation=cv2.INTER_CUBIC)
            now_std = now_std + 1
        # print(stdimg_path)
        # print(img_path)
        imgnow = cv2.imread(img_path, 0)
        ssim = calculate_ssim(imgnow, stdimg)
        psnr = calculate_psnr(imgnow, stdimg)
        imnow = trans(imgnow).cuda()
        stdim = trans(stdimg).cuda()
        # lp = calculate_lpips(imnow, stdim).cpu().detach().numpy()
        lp = loss_fn.forward(imnow, stdim).cpu().detach().numpy()
        ssims.append(ssim)
        psnrs.append(psnr)
        lpipses.append(lp)
        i = i+1
    
    return ssims, psnrs, lpipses



if __name__ == '__main__':
    testset_dirs = [
        '/home/ubuntu/sharedData/YYK/SICE_test/std/'
    ]
    MY = [opt.data_path+'low/', opt.data_path+'over/']
    
    MSEC = [
        '/home/ubuntu/sharedData/YYK/Myenhance/test/MSEC/',
        '/home/ubuntu/sharedData/YYK/Myenhance/test/MSEC/'
        ]

    for i in range(2):
        ssims, psnrs, lpipses = Test_train(testset_dirs[0], MY[i])
        # Test_train(testset_dirs[i], PEC[i])
        if i == 0:
            under_ssims, under_psnrs, under_lpipses = ssims, psnrs, lpipses
        else:
            over_ssims, over_psnrs, over_lpipses = ssims, psnrs, lpipses

    print('AVG_under_ssims', sum(under_ssims)/len(under_ssims))
    print('AVG_under_psnrs', sum(under_psnrs)/len(under_psnrs))
    print('AVG_under_lpips', sum(under_lpipses)/len(under_lpipses))
    print('AVG_over_ssims', sum(over_ssims)/len(over_ssims))
    print('AVG_over_psnrs', sum(over_psnrs)/len(over_psnrs))
    print('AVG_over_lpips', sum(over_lpipses)/len(over_lpipses))
    print('AVG_ssims', (sum(over_ssims) + sum(under_ssims))/(len(over_ssims) + len(under_ssims)))
    print('AVG_psnrs', (sum(over_psnrs) + sum(under_psnrs))/(len(over_psnrs) + len(under_psnrs)))
    print('AVG_lpips', (sum(over_lpipses) + sum(under_lpipses))/(len(over_lpipses) + len(under_lpipses)))

    content = [
        'under_ssims' + str(sum(under_ssims)/len(under_ssims)),
        'under_psnrs' + str(sum(under_psnrs)/len(under_psnrs)),
        'under_lpipses' + str(sum(under_lpipses)/len(under_lpipses)),
        'over_ssims' + str(sum(over_ssims)/len(over_ssims)),
        'over_psnrs' + str(sum(over_psnrs)/len(over_psnrs)),
        'over_lpipses' + str(sum(over_lpipses)/len(over_lpipses)),
        'all_ssims' + str((sum(over_ssims) + sum(under_ssims))/(len(over_ssims) + len(under_ssims))),
        'all_psnrs' + str((sum(over_psnrs) + sum(under_psnrs))/(len(over_psnrs) + len(under_psnrs))),
        'all_lpipses' + str((sum(over_lpipses) + sum(under_lpipses))/(len(over_lpipses) + len(under_lpipses))),
    ]

    ### Use .txt to save quantative result
    # set name of .txt
    txtpath = './test/test_results/' + 'SICE_' + opt.tail
    filename = f"{txtpath}.txt"
    suffix = 0

    # save .txt
    while os.path.isfile(filename):
        suffix += 1
        filename = f"{txtpath}_{suffix}.txt"

    # write .txt
    with open(filename, "w") as f:
        for line in content:
            f.write(line + "\n")