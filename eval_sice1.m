clear;

% delete(gcp('nocreate'));
% p = parpool('local',20);

addpath('niqe/');
load modelparameters.mat
blocksizerow    = 96;
blocksizecol    = 96;
blockrowoverlap = 0;
blockcoloverlap = 0;

%%% results paths
% e1 = '/home/ubuntu/sharedData/YYK/Myenhance/test/KinD_p-sice/';
% e2 = '/home/ubuntu/sharedData/YYK/Myenhance/test/Uretinex-sice/';
% e3 = '/home/ubuntu/sharedData/YYK/Myenhance/test/Zero-DCE-sice/';
% e4 = '/home/ubuntu/sharedData/YYK/Myenhance/test/RUAS-sice/';
% e5 = '/home/ubuntu/sharedData/YYK/Myenhance/test/SCI_easy-sice/';
% e6 = '/home/ubuntu/sharedData/YYK/Myenhance/test/SCI_medium-sice/';
% e7 = '/home/ubuntu/sharedData/YYK/Myenhance/test/SCI_difficult-sice/';
% e8 = '/home/ubuntu/sharedData/YYK/Myenhance/test/MSEC-sice/';
% e9 = '/home/ubuntu/sharedData/YYK/Myenhance/test/FEC-sice/';
% e10 = '/home/ubuntu/sharedData/YYK/Myenhance/test/LCD-sice/';
% e11 = '/home/ubuntu/sharedData/YYK/Myenhance/test/PEC-sice/';
% e12 = '/home/ubuntu/sharedData/YYK/Myenhance/test/SICE_Stkm0_15/';
% e12 = '/home/ubuntu/sharedData/YYK/Myenhance/test/SICE_Stkm3_14/';
e12 = '/home/ubuntu/sharedData/YYK/Myenhance_k6/test/SICE_Stkm3_18/';
enhance_imgDirs = {e12};


all_under_niqe = [];
all_over_niqe = [];
all_niqe = [];

for all_i = 1:length(enhance_imgDirs)

enhance_imgDir = enhance_imgDirs{all_i};
disp(enhance_imgDir);

oldPwd = pwd;  

under_niqe = [];
over_niqe = [];
d1 = 'low/';
d2 = 'over/';
d = {d1, d2};

for nameid = 1:2
nowd = d{nameid};
% disp(nowd);
now_enhance_imgDir = strcat(enhance_imgDir, nowd);
disp(now_enhance_imgDir);

cd(now_enhance_imgDir);  
x = dir;  
list_Of_enhance_Images = [];  
for i = 1:length(x)
    if x(i).isdir == 0
          list_Of_enhance_Images = [list_Of_enhance_Images; x(i)]; 
          % namenow = x(i).name;
          % disp(namenow(1:3))
    end
end
cd(oldPwd);  

num = 1;
for i = 1:length(list_Of_enhance_Images)
    enhance_img_path = [list_Of_enhance_Images(i).folder, '/', list_Of_enhance_Images(i).name];
    enhance_img = imread(enhance_img_path);
    niqe = computequality(enhance_img,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap,mu_prisparam,cov_prisparam);
    % disp(num);
    % disp(niqe);
    if length(nowd) == 4
        under_niqe = [under_niqe; niqe];
    end
    if length(nowd) == 5
        over_niqe = [over_niqe; niqe];
    end
    num = num + 1;
end

end

disp(enhance_imgDir);
disp('under_niqe ');
disp(sum(under_niqe) / length(under_niqe));
disp('over_niqe ');
disp(sum(over_niqe) / length(over_niqe));
disp('all_niqe ');
disp( (sum(under_niqe) + sum(over_niqe)) / (length(under_niqe) + length(over_niqe)) )

all_under_niqe = [all_under_niqe;sum(under_niqe) / length(under_niqe)];
all_over_niqe = [all_over_niqe;sum(over_niqe) / length(over_niqe)];
all_niqe = [all_niqe;(sum(under_niqe) + sum(over_niqe)) / (length(under_niqe) + length(over_niqe))];

end