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

def getparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_cpus", type=int, default=32)
    parser.add_argument("--data_path", type=str, default='./test/EED_StKm3_8/')
    parser.add_argument("--tail", type=str, default='MY')
    # parser.add_argument("--tail", type=str, default='KinD_p')
    parser.add_argument("--test_my", type=bool, default=True)   ### Here is used to judge whether to use our results or others  

    print(parser.parse_args())
    return parser.parse_args()

opt = getparser()
under_ssims = []
under_psnrs = []
under_niqe = []
under_lpips = []
over_lpips = []
over_niqe = []
over_ssims = []
over_psnrs = []
all_psnrs = []
all_lpipss = []
all_ssims = []

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

    Under_ssim = []
    Over_ssim = []
    Under_psnr = []
    Over_psnr = []
    Under_lpips = []
    Over_lpips = []

    i = 0
    stdimg = None
    stdimg_path = ' '
    for img in tqdm(images):
        img_path = test_dir + img
        new_stdimg_path = std_imgs_path + stdimages[int(i/5)]
        if not new_stdimg_path == stdimg_path:
            stdimg_path = new_stdimg_path
            stdimg = cv2.imread(stdimg_path, 0)
        imgnow = cv2.imread(img_path, 0)
        if i % 5 == 1 or i % 5 == 2:
            ssim = calculate_ssim(imgnow, stdimg)
            psnr = calculate_psnr(imgnow, stdimg)
            imnow = trans(imgnow).cuda()
            stdim = trans(stdimg).cuda()
            lp = loss_fn.forward(imnow, stdim).cpu().detach().numpy()
            Under_ssim.append(ssim)
            Under_psnr.append(psnr)
            Under_lpips.append(lp)
        elif i % 5 == 3 or i % 5 == 4 or i % 5 == 0:
            ssim = calculate_ssim(imgnow, stdimg)
            psnr = calculate_psnr(imgnow, stdimg)
            imnow = trans(imgnow).cuda()
            stdim = trans(stdimg).cuda()
            lp = loss_fn.forward(imnow, stdim).cpu().detach().numpy()
            Over_ssim.append(ssim)
            Over_psnr.append(psnr)
            Over_lpips.append(lp)
        i = i+1
    
    under_ssims.append(sum(Under_ssim)/len(Under_ssim))
    under_psnrs.append(sum(Under_psnr)/len(Under_psnr))
    under_lpips.append(sum(Under_lpips)/len(Under_lpips))

    over_ssims.append(sum(Over_ssim)/len(Over_ssim))
    over_psnrs.append(sum(Over_psnr)/len(Over_psnr))
    over_lpips.append(sum(Over_lpips)/len(Over_lpips))
    
    print('Underexposure\'s SSIM: ', sum(Under_ssim)/len(Under_ssim))
    print('Underexposure\'s PSNR: ', sum(Under_psnr)/len(Under_psnr))
    print('Underexposure\'s LPIPS: ', sum(Under_lpips)/len(Under_lpips))
    print('Overexposure\'s SSIM:   ', sum(Over_ssim)/len(Over_ssim))
    print('Overexposure\'s PSNR:   ', sum(Over_psnr)/len(Over_psnr))
    print('Overexposure\'s LPIPS:  ', sum(Over_lpips)/len(Over_lpips))

    all_ssims.append( (sum(Under_ssim)+sum(Over_ssim))/(len(Under_ssim)+len(Over_ssim)) )
    all_psnrs.append( (sum(Under_psnr)+sum(Over_psnr))/(len(Under_psnr)+len(Over_psnr)) )
    all_lpipss.append( (sum(Under_lpips)+sum(Over_lpips))/(len(Under_lpips)+len(Over_lpips)) )
    
    print('Allexposure\'s SSIM:   ', (sum(Under_ssim)+sum(Over_ssim))/(len(Under_ssim)+len(Over_ssim)))
    print('Allexposure\'s PSNR:   ', (sum(Under_psnr)+sum(Over_psnr))/(len(Under_psnr)+len(Over_psnr)))
    print('Allexposure\'s LPIPS:  ', (sum(Under_lpips)+sum(Over_lpips))/(len(Under_lpips)+len(Over_lpips)))



if __name__ == '__main__':
    ### Here are standard images' path
    testset_dirs = [
        '/home/ubuntu/sharedData/YYK/EED/testing/expert_a_testing_set/',
        '/home/ubuntu/sharedData/YYK/EED/testing/expert_b_testing_set/',
        '/home/ubuntu/sharedData/YYK/EED/testing/expert_c_testing_set/',
        '/home/ubuntu/sharedData/YYK/EED/testing/expert_d_testing_set/',
        '/home/ubuntu/sharedData/YYK/EED/testing/expert_e_testing_set/'
    ]
    ### Here are UREC images' path
    MY = [opt.data_path+'a/', opt.data_path+'b/', opt.data_path+'c/', opt.data_path+'d/', opt.data_path+'e/']
    
    ### Here are other methods' images' path
    MSEC = '/home/ubuntu/sharedData/YYK/Myenhance/test/MSEC/'
    SCI_m = '/home/ubuntu/sharedData/YYK/Myenhance/test/SCI_medium/'
    SCI_d = '/home/ubuntu/sharedData/YYK/Myenhance/test/SCI_difficult/'
    SCI_e = '/home/ubuntu/sharedData/YYK/Myenhance/test/SCI_easy/'
    RUAS = '/home/ubuntu/sharedData/YYK/Myenhance/test/RUAS/'
    Zero_DCE = '/home/ubuntu/sharedData/YYK/Myenhance/test/Zero-DCE/'
    URetinex = '/home/ubuntu/sharedData/YYK/Myenhance/test/Uretinex/'
    PEC = '/home/ubuntu/sharedData/YYK/Myenhance/test/PEC/'
    LCD = '/home/ubuntu/sharedData/YYK/Myenhance/test/LCD/'
    FEC = '/home/ubuntu/sharedData/YYK/Myenhance/test/FEC/'
    KinD_p = '/home/ubuntu/sharedData/YYK/Myenhance/test/KinD_p/'

    for i in range(5):
        if opt.test_my:
            Test_train(testset_dirs[i], MY[i])
        else:
            Test_train(testset_dirs[i], KinD_p)

    print('under_ssims', under_ssims)
    print('under_psnrs', under_psnrs)
    print('under_lpips', under_lpips)
    print('over_ssims', over_ssims)
    print('over_psnrs', over_psnrs)
    print('over_lpips', over_lpips)

    print('AVG_under_ssims', sum(under_ssims)/len(under_ssims))
    print('AVG_under_psnrs', sum(under_psnrs)/len(under_psnrs))
    print('AVG_under_lpips', sum(under_lpips)/len(under_lpips))
    print('AVG_over_ssims', sum(over_ssims)/len(over_ssims))
    print('AVG_over_psnrs', sum(over_psnrs)/len(over_psnrs))
    print('AVG_over_lpips', sum(over_lpips)/len(over_lpips))
    print('AVG_all_ssims', sum(all_ssims)/len(all_ssims))
    print('AVG_all_psnrs', sum(all_psnrs)/len(all_psnrs))
    print('AVG_all_lpips', sum(all_lpipss)/len(all_lpipss))

    content = [
        'under_ssims' + str(under_ssims),
        'under_psnrs' + str(under_psnrs),
        'under_lpips' + str(under_lpips),
        'over_ssims' + str(over_ssims),
        'over_psnrs' + str(over_psnrs),
        'over_lpips' + str(over_lpips),
        'AVG_under_ssims ' + str(sum(under_ssims)/len(under_ssims)),
        'AVG_under_psnrs ' + str(sum(under_psnrs)/len(under_psnrs)),
        'AVG_under_lpips ' + str(sum(under_lpips)/len(under_lpips)),
        'AVG_over_ssims  ' + str(sum(over_ssims)/len(over_ssims)),
        'AVG_over_psnrs  ' + str(sum(over_psnrs)/len(over_psnrs)),
        'AVG_over_lpips  ' + str(sum(over_lpips)/len(over_lpips)),
        'AVG_all_ssims  ' + str(sum(all_ssims)/len(all_ssims)),
        'AVG_all_psnrs  ' + str(sum(all_psnrs)/len(all_psnrs)),
        'AVG_all_lpips  ' + str(sum(all_lpipss)/len(all_lpipss)),
    ]



    ### Use .txt to save quantative result
    # set name of .txt
    txtpath = './test/test_results/' + 'EED_' + opt.tail
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