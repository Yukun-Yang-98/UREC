from util.util import get_dir_name
from util.guided_filter import *
from dataset.dataset import ImageDataset
from model.guidemodel import *
from model.reRmodel import *

import torchvision.transforms.functional as F
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import shutil
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision

def getparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_cpus", type=int, default=32)
    parser.add_argument("--data_path", type=str, default='/home/ubuntu/sharedData/YYK/MY_RESEARCH/UREC/Myenhance_new/test_imgs/')
    parser.add_argument("--model", type=str, default='3d')
    parser.add_argument("--save_img", type=bool, default=True)
    parser.add_argument("--img_size", type=int, default=[1200, 900])
    parser.add_argument("--resize", type=bool, default=False)
    parser.add_argument("--save_path", type=str, default='./test/')
    parser.add_argument("--save_dir", type=str, default='test_')
    print(parser.parse_args())
    return parser.parse_args()

opt = getparser()

checkpoint_gm = torch.load('./curve.pth')   #small
if opt.model == '1d':
    checkpoint_1d = torch.load('./Decom_1d.pth')
    checkpoint3_1d = torch.load('./Expcontrol_1d.pth')
elif opt.model == '3d':
    checkpoint_3d = torch.load('./Decom_3d.pth')
    checkpoint3_3d = torch.load('./Expcontrol_3d.pth')

img_path, gm_path, k_path = '', '', ''
if opt.save_img:
    run_dir = get_dir_name(opt.save_path, opt.save_dir)
    os.makedirs(run_dir)
    os.makedirs(run_dir + '/save_files/')
    img_path = run_dir + '/UREC/'
    gm_path = run_dir + '/gm_files/'
    os.makedirs(img_path)
    os.makedirs(gm_path)
    shutil.copyfile('./test.py', run_dir + '/save_files/' + 'test.py')
    test_dirs = [run_dir]

# def gen_guidemap(img, S=0.5, A1=None, A2=None):
#     '''  This is the method we mentioned in the paper for generating the guidance map used for the EED and SICE datasets, where the value of S is the mean of the ground truth image.  '''
#     img = torch.mean(img, dim=1, keepdim=True)
#     guide_map = torch.ones_like(img)*S
#     return guide_map

def gen_guidemap(img, S=0.5, A1=-10, A2=0.01):
    '''  This is the method mentioned in the paper for generating the guidance map manually.  '''
    ep = 1e-6
    A1_thr = -80
    A2_thr = 80
    if A1_thr < A1:
        a1 = S/(np.exp(A1*S)-1)
        b1 = -1*a1
    else:
        a1 = 200/(A1+100+ep)
    if A2 < A2_thr:
        a2 = (S-1)/(np.exp(A2*S)-np.exp(A2)+ep)
        b2 = 1-a2*np.exp(A2)
    else:
        a2 = -200/(A2-100-ep)
    img = torch.mean(img, dim=1, keepdim=True)
    # img = torch.max(img, dim=1, keepdim=True)[0].data
    mask_low = (img<=S)
    mask_over = (img>S)
    guide_map = torch.zeros_like(img)
    for i in range(2):
        img_copy = img.clone()
        if i == 1:
            if A1_thr < A1:
                if A1 == 0:
                    img_copy = torch.ones_like(img_copy)*S
                else:
                    img_copy = a1*torch.exp(A1*img_copy) + b1
                # img_copy = a1*torch.exp(A1*img_copy) + b1
            else:
                img_copy = a1*img_copy
                img_copy = torch.where(img_copy > S, S, img_copy)
            img_copy[~mask_low] = 0
            guide_map = guide_map + img_copy
        else:
            if A2 < A2_thr:
                if A2 == 0:
                    img_copy = torch.ones_like(img_copy)*S
                else:
                    img_copy = a2*torch.exp(A2*img_copy) + b2
                # img_copy = a2*torch.exp(A2*img_copy) + b2
            else:
                img_copy = a2*img_copy-a2+1
                img_copy = torch.where(img_copy < S, S, img_copy)
            img_copy[~mask_over] = 0
            guide_map = guide_map + img_copy
    return guide_map

def save_img_batch(batch, dirpath, imgname, save_num=1):
        imgpath = dirpath + '/' + imgname
        assert len(batch.shape) == 4
        torchvision.utils.save_image(batch[:save_num], imgpath)

class K_fit_model(nn.Module):
    ''' To inference a Kmap for input image generating a Rout with illuminance settde by guidemap '''
    def __init__(self, model='3d'):
        super(K_fit_model, self).__init__()
        if model =='3d':
            self.kmodel = Km0_net_3d()
        elif model =='1d':
            self.kmodel = Km0_net_1d()
    def forward(self, I, R, Rm, guide_map):
        Kmap = self.kmodel(I, R, Rm, guide_map)
        return Kmap

class Curve_model(nn.Module):
    ''' To inference a Kmap for input image generating a Rout with illuminance settde by guidemap '''
    def __init__(self):
        super(Curve_model, self).__init__()
        self.points = 3
        self.model = Guidemap_auto_net_sm(points=self.points)
    def forward(self, I):
        Kmap, Amap = self.model(I)
        Ps = torch.mean(Kmap, dim=(2, 3))
        XYs = torch.split(Ps, self.points, dim=1)
        Xs = XYs[0]
        Ys = XYs[1]
        Xs, _ = torch.sort(Xs, dim=1)
        Ys, _ = torch.sort(Ys, dim=1)
        As = torch.mean(Amap, dim=(2, 3))
        As = (As-0.5)*80
        As[As==0] = 1e-6
        return Xs, Ys, As

def Test_train(S=0.45, A1=0, A2=0, AUTO=False):
    '''  AUTO: -True- Use the automatically generated guidance map method described in the appendix.  '''
    '''  AUTO: -False- generated guided map manually.  '''
    if opt.model == '1d':
        decmodel = Decom_1d()
        Kfitmodel = K_fit_model(model=opt.model)
        checkpoint = checkpoint_1d
        checkpoint3 = checkpoint3_1d
    elif opt.model == '3d':
        decmodel = Decom_U3_L_sm()
        Kfitmodel = K_fit_model(model=opt.model)
        checkpoint = checkpoint_3d
        checkpoint3 = checkpoint3_3d
    
    state_dict =checkpoint['Decom']
    state_dict3 =checkpoint3['StageK2']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if k[0] == 'm':
            name = k[7:] # remove 'module.' of dataparallel
        new_state_dict[name]=v
    decmodel.load_state_dict(new_state_dict)
    new_state_dict = OrderedDict()
    for k, v in state_dict3.items():
        name = k
        if k[0] == 'm':
            name = k[7:] # remove 'module.' of dataparallel
        new_state_dict[name]=v
    Kfitmodel.load_state_dict(new_state_dict)

    cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    if cuda:
        decmodel.cuda()
        Kfitmodel.cuda()

    if AUTO:
        Gm_model = Curve_model()
        checkpoint = checkpoint_gm
        state_dict =checkpoint['StageK2']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k
            if k[0] == 'm':
                name = k[7:] # remove 'module.' of dataparallel
            new_state_dict[name]=v
        Gm_model.load_state_dict(new_state_dict)
        Gm_model.cuda()

    if opt.resize:
        transforms_ = [
            transforms.Resize([opt.img_size[1], opt.img_size[0]], Image.BICUBIC),
            transforms.ToTensor(),
        ]
    else:
        transforms_ = [
            transforms.ToTensor(),
        ]

    test_imgs = os.listdir(opt.data_path)
    test_imgs.sort()
    dataloader = DataLoader(
        ImageDataset(opt.data_path, transform_=transforms_),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpus,
    )
    print(len(dataloader))

    now = 0
    numbatches = len(dataloader)
    decmodel.eval()
    Kfitmodel.eval()
    with torch.no_grad():
        for epoch in range(0, opt.epochs):
            pbar = enumerate(dataloader)
            pbar = tqdm(pbar, total=numbatches)
            # std_illuminance = 0.4
            for i, batch in pbar:
                # set model input
                input = Variable(batch['img'].type(Tensor))
                input_mean = torch.mean(input, dim=1, keepdim=True)
                if AUTO:
                    Xs, Ys, As = Gm_model(input_mean)
                    guide_map = curve_change(input_mean, Xs, Ys, As)

                if opt.model == '1d':
                    R0, _ = decmodel(input_mean)
                    R0m, _ = decmodel(1-input_mean)
                    if not AUTO:
                        guide_map = gen_guidemap(input, S, A1, A2)
                    K_fit_map = Kfitmodel(input_mean, R0, R0m, guide_map)
                    K_fit_map = guidedfilter2d_gray(input_mean, K_fit_map)
                    K0 = K_fit_map[:, 0:1, :, :]
                    Km = K_fit_map[:, 1:2, :, :]
                    Rout = input_mean + K0*R0 - Km*R0m
                    Iout = input * Rout / (input_mean+1e-6)
                    Iout = torch.clamp(Iout, 0, 1)

                elif opt.model == '3d':
                    R0, _ = decmodel(input)
                    R0m, _ = decmodel(1-input)
                    if not AUTO:
                        guide_map = gen_guidemap(input, S, A1, A2)
                    K_fit_map = Kfitmodel(input, R0, R0m, guide_map)
                    K_fit_map = guidedfilter2d_gray(input_mean, K_fit_map)
                    K0 = K_fit_map[:, 0:1, :, :]
                    Km = K_fit_map[:, 1:2, :, :]
                    Rout = input + K0*R0 - Km*R0m
                    Rout = torch.clamp(Rout, 0, 1)
                    Iout = Rout


                now += 1
                if opt.save_img:
                    img_name = test_imgs[i]
                    save_img_batch(Iout, img_path, img_name)
                    img_name = test_imgs[i].split('.')[0]
                    img_tail = test_imgs[i].split('.')[1]
                    save_img_batch(guide_map, gm_path, img_name+'_gm'+'.'+img_tail)
                
            print("======== epoch " + str(epoch) + " has been finished ========")
            if opt.save_img:
                print('run dir is:   ', run_dir)
    

if __name__ == '__main__':
    Test_train(S=0.55, A1=0, A2=0, AUTO=True)
    # Test_train(S=0.55, A1=5, A2=-5, AUTO=False)