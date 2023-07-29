from dataset.dataset import *
from model.reRmodel import *
from util.guided_filter import *
from util.util import *

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


def Norm(data):
    B, C, W, H = data.shape
    out = []
    for i in range(0, B):
        data_piece = data[i:i+1]
        avg_v = torch.mean(data_piece)
        data_piece = data_piece - avg_v
        min_v = torch.min(data_piece)
        max_v = torch.max(data_piece)
        diff = max_v - min_v
        data_piece = (data_piece-min_v) / diff
        data_piece = (data_piece-0.5).true_divide(0.5)
        if out == []:
            out = data_piece
        else:
            out = torch.cat((out, data_piece), dim=0)
    return out

def gen_guidemap(img, S, A, rout=None):
    L = torch.mean(img, dim=1, keepdim=True)
    Lnorm = Norm(L)
    guide_map = L - A*(Lnorm)
    for i in range(10):
        guide_map = torch.clamp(guide_map, 0, 1)
        guide_map = guide_map*(S/torch.mean(guide_map))
    return guide_map

def getparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_cpus", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=[1200, 900])
    parser.add_argument("--save_img", type=bool, default=True)
    parser.add_argument("--resize", type=bool, default=True)

    print(parser.parse_args())
    return parser.parse_args()

opt = getparser()

'''  Here is the path of SICE input/standard images  '''
std_imgs_path = '/home/ubuntu/sharedData/YYK/SICE_test/std/'
test_dirs = [
    '/home/ubuntu/sharedData/YYK/SICE_test/over/',
    '/home/ubuntu/sharedData/YYK/SICE_test/low/',
]

### The path of weights need to be put here, checkpoint1 for Retinex decomposition network, checkpoint2 for exposure control network.
checkpoint1 = torch.load('./run/Decom_L36/save_files/5000DecomRL.pth')
checkpoint2 = torch.load('./run/Stage_Km3_12/save_files/200StageKm0.pth')

if opt.save_img:
    run_dir = get_dir_name('./test', 'SICE_test_')
    os.makedirs(run_dir)
    os.makedirs(run_dir + '/save_files/')
    shutil.copyfile('./test_sice.py', run_dir + '/save_files/' + 'test_sice.py')
    testdir_low = run_dir + '/low'
    testdir_over = run_dir + '/over'
    os.makedirs(testdir_low)
    os.makedirs(testdir_over)
    out_dirs = [testdir_over, testdir_low]

class K_fit_model(nn.Module):
    ''' To inference a Kmap for input image generating a Rout with illuminance settde by guidemap '''
    def __init__(self):
        super(K_fit_model, self).__init__()
        self.kmodel = Km0_net()

    def forward(self, I, R, Rm, S, guide_map):
        Smap = torch.ones_like(I)
        Smap = Smap * S
        Kmap = self.kmodel(I, R, Rm, Smap, guide_map)
        return Kmap    

def sample_img_sice(i, img, name, dir):
    unloader = transform.ToPILImage()

    input = img
    num = str(i)
    while (len(num) < 4):
        num = "0" + num
    input_name = dir + '/' + name + ".jpg"
    input = input.cpu().detach().numpy()
    input = np.clip(input * 255.0, 0, 255).astype(np.uint8)
    img = unloader(input.transpose([1, 2, 0]))
    img.save(input_name)

def Test_train(std_imgs_path, test_dir, output_dir, alpha=0):
    decmodel = Decom_U3_ReLU()
    Kfitmodel = K_fit_model()

    state_dict =checkpoint1['Decom']
    state_dict3 =checkpoint2['StageK2']
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

    if opt.resize:
        trans_std = transforms.Compose([
            transforms.Resize([opt.img_size[1], opt.img_size[0]], Image.BICUBIC),
            transforms.ToTensor(),
        ])
        transforms_ = [
            transforms.Resize([opt.img_size[1], opt.img_size[0]], Image.BICUBIC),
            transforms.ToTensor(),
        ]
    else:
        transforms_ = [
            transforms.ToTensor(),
        ]
        trans_std = transforms.Compose([
            transforms.ToTensor(),
        ])

    std_imgs = os.listdir(std_imgs_path)
    std_imgs.sort()
    test_imgs = os.listdir(test_dir)
    test_imgs.sort()
    dataloader = DataLoader(
        ImageDataset(test_dir, transform_=transforms_),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpus,
    )
    print(len(dataloader))

    now = 0
    numbatches = len(dataloader)

    decmodel.eval()
    Kfitmodel.eval()
    img_f3 = ''

    with torch.no_grad():
        for epoch in range(0, opt.epochs):
            pbar = enumerate(dataloader)
            pbar = tqdm(pbar, total=numbatches)
            std_illuminance = 0.5
            now_std = 0
            for i, batch in pbar:
                # set model input
                input = Variable(batch['img'].type(Tensor))
                input_mean = torch.mean(input, dim=1, keepdim=True)
                input_mean_minus = torch.mean(1-input, dim=1, keepdim=True)
                # Train
                img_f3_now = test_imgs[i][0:3]
                if not img_f3 == img_f3_now:
                    img_f3 = img_f3_now
                    std_img = Image.open(std_imgs_path+std_imgs[now_std])
                    std_img = trans_std(std_img)
                    std_illuminance = torch.mean(std_img)
                    now_std = now_std + 1
                R0, _ = decmodel(input)
                R0m, _ = decmodel(1-input)

                S = std_illuminance
                A = alpha*S

                guide_map = gen_guidemap(input, S, A)

                K_fit_map = Kfitmodel(input, R0, R0m, S, guide_map)

                K0 = K_fit_map[:, 0:1, :, :]   ### K0 represents K_{I}
                Km = K_fit_map[:, 1:2, :, :]   ### Km represents K_{1-I}

                K0 = guidedfilter2d_gray(input_mean, K0)
                Km = guidedfilter2d_gray(input_mean_minus, Km)

                Rout = input + K0*R0 - Km*R0m

                Rout = torch.clamp(Rout, 0, 1)

                now += 1
                if opt.save_img:
                    S = np.around(S.cpu().detach().numpy(), 4)
                    A = np.around(A.cpu().detach().numpy(), 4)
                    avg_im = np.around(torch.mean(Rout[0, :, :, :]).cpu().detach().numpy(), 4)
                    # sample_img_EED(now, input[0, :, :, :], name=str(i)+'std', dir=run_dir)
                    sample_img_sice(i, Rout[0, :, :, :], test_imgs[i] + "_avg:"+str(avg_im) + '_S:'+str(S), dir=output_dir)
                    # sample_img_sice(i, guide_map[0, :, :, :], test_imgs[i] + 'gm' + '_S:'+str(S) + '_A:'+str(A), dir=output_dir)
                
            print("======== epoch " + str(epoch) + " has been finished ========")
            if opt.save_img:
                print('run dir is:   ', run_dir)


if __name__ == '__main__':
    alpha = 0
    for i in range(2):
        if i==0:
            alpha = 0
        else:
            alpha = 0
        Test_train(std_imgs_path=std_imgs_path, test_dir=test_dirs[i], output_dir=out_dirs[i], alpha=alpha)
