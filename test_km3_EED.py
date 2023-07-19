from dataset.dataset import *
from model.reRmodel import *
from util.guided_filter import *

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


def getparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_cpus", type=int, default=32)
    parser.add_argument("--data_path", type=str, default='/home/ubuntu/sharedData/YYK/EED/testing/INPUT_IMAGES/')  ### Here is the path of input images of EED
    parser.add_argument("--save_img", type=bool, default=True)
    parser.add_argument("--resize", type=bool, default=False)

    print(parser.parse_args())
    return parser.parse_args()

### The path of weights need to be put here, checkpoint1 for Retinex decomposition network, checkpoint2 for exposure control network.
checkpoint1 = torch.load('./run/Decom_L37/save_files/5000DecomRL.pth')
checkpoint2 = torch.load('./run/Stage_Km3_15/save_files/200StageKm0.pth')

opt = getparser()
### Here is the path of standard images of EED, the standard images are used to calculate preset luminance. (preset luminance = torch.mean(std image)) 
testset_dirs = [
    '/home/ubuntu/sharedData/YYK/EED/testing/expert_a_testing_set/',
    '/home/ubuntu/sharedData/YYK/EED/testing/expert_b_testing_set/',
    '/home/ubuntu/sharedData/YYK/EED/testing/expert_c_testing_set/',
    '/home/ubuntu/sharedData/YYK/EED/testing/expert_d_testing_set/',
    '/home/ubuntu/sharedData/YYK/EED/testing/expert_e_testing_set/'
]
if opt.save_img:
    run_dir = get_dir_name('./test', 'EED_StKm3_')
    os.makedirs(run_dir)
    os.makedirs(run_dir + '/save_files/')
    shutil.copyfile('./test_km3_EED.py', run_dir + '/save_files/' + 'test_km3_EED.py')
    testdira = run_dir + '/a'
    testdirb = run_dir + '/b'
    testdirc = run_dir + '/c'
    testdird = run_dir + '/d'
    testdire = run_dir + '/e'
    os.makedirs(testdira)
    os.makedirs(testdirb)
    os.makedirs(testdirc)
    os.makedirs(testdird)
    os.makedirs(testdire)
    test_dirs = [testdira, testdirb, testdirc, testdird, testdire]


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

def gen_guidemap(img, S, A, rout=None):
    L = torch.mean(img, dim=1, keepdim=True)
    Lnorm = Norm(L)
    guide_map = L - A*(Lnorm)
    for i in range(10):
        guide_map = torch.clamp(guide_map, 0, 1)
        guide_map = guide_map*(S/torch.mean(guide_map))
    return guide_map

def Norm(data):
    B, C, W, H = data.shape
    out = []
    for i in range(0, B):
        data_piece = data[i:i+1]
        # print(data_piece.shape)
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

def sample_img_EED(i, img, name, dir):
    unloader = transform.ToPILImage()

    input = img
    num = str(i)
    while (len(num) < 4):
        num = "0" + num
    input_name = dir + '/' + num + "_" + name + ".jpg"
    input = input.cpu().detach().numpy()
    input = np.clip(input * 255.0, 0, 255).astype(np.uint8)
    img = unloader(input.transpose([1, 2, 0]))
    img.save(input_name)


def Test_train(std_imgs_path, output_dir):

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
            transforms.Resize(int(opt.img_size[0]), Image.BICUBIC),
            transforms.ToTensor(),
        ])
        transforms_ = [
            transforms.Resize(int(opt.img_size[0]), Image.BICUBIC),
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
            std_illuminance = 0.5
            for i, batch in pbar:
                # set model input
                input = Variable(batch['img'].type(Tensor))
                input_mean = torch.mean(input, dim=1, keepdim=True)
                input_mean_minus = torch.mean(1-input, dim=1, keepdim=True)

                std_img = Image.open(std_imgs_path+std_imgs[int(i/5)])
                std_img = trans_std(std_img)
                std_illuminance = torch.mean(std_img)
                R0, L0 = decmodel(input)
                R0m, _ = decmodel(1-input)
                S = std_illuminance
                A = 0

                guide_map = gen_guidemap(input, S, A)
                
                K_fit_map = Kfitmodel(input, R0, R0m, S, guide_map)

                K0 = K_fit_map[:, 0:1, :, :]
                Km = K_fit_map[:, 1:2, :, :]
                K0_g = guidedfilter2d_gray(input_mean, K0)
                Km_g = guidedfilter2d_gray(input_mean_minus, Km)

                Rout = input + K0_g*R0 - Km_g*R0m

                Rout = torch.clamp(Rout, 0, 1)
                
                now += 1
                if opt.save_img:
                    S = np.around(S.cpu().detach().numpy(), 4)
                    # A = np.around(A.cpu().detach().numpy(), 4)
                    avg_gm = np.around(torch.mean(guide_map[0, :, :, :]).cpu().detach().numpy(), 4)
                    avg_im = np.around(torch.mean(Rout[0, :, :, :]).cpu().detach().numpy(), 4)
                    # sample_img_EED(now, input[0, :, :, :], name=str(i)+'std', dir=run_dir)
                    sample_img_EED(i, Rout[0, :, :, :], "avg:"+str(avg_im) + '_S:'+str(S), dir=output_dir)
                    # sample_single_img(i, guide_map[0, :, :, :], name='gm' + '_S:'+str(S) + '_A:'+str(A) + '_avg:'+str(avg_gm), dir=output_dir)
                
            print("======== epoch " + str(epoch) + " has been finished ========")
            if opt.save_img:
                print('run dir is:   ', run_dir)
    


if __name__ == '__main__':
    for i in range(5):
        Test_train(std_imgs_path=testset_dirs[i], output_dir=test_dirs[i])
