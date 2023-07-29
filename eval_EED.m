% delete(gcp('nocreate'));
% p = parpool('local',20);

addpath('niqe/');
load modelparameters.mat
blocksizerow    = 96;
blocksizecol    = 96;
blockrowoverlap = 0;
blockcoloverlap = 0;

input_imgDir = '/home/ubuntu/sharedData/YYK/EED/testing/INPUT_IMAGES/';

%%% UREC results path
enhance_imgDir = ["EED results path/a/";
    "EED results path/b/";
    "EED results path/c/";
    "EED results path/d/";
    "EED results path/e/"];

%%% other results path
% enhance_imgDir = ["/home/ubuntu/sharedData/YYK/Myenhance/test/MSEC/"];
% enhance_imgDir = ["/home/ubuntu/sharedData/YYK/Myenhance/test/SCI_medium/"];
% enhance_imgDir = ["/home/ubuntu/sharedData/YYK/Myenhance/test/SCI_easy/"];
% enhance_imgDir = ["/home/ubuntu/sharedData/YYK/Myenhance/test/SCI_difficult/"];
% enhance_imgDir = ["/home/ubuntu/sharedData/YYK/Myenhance/test/FEC/"];
% enhance_imgDir = ["/home/ubuntu/sharedData/YYK/Myenhance/test/LCD/"];
% enhance_imgDir = ["/home/ubuntu/sharedData/YYK/Myenhance/test/RUAS/"];
% enhance_imgDir = ["/home/ubuntu/sharedData/YYK/Myenhance/test/Uretinex/"];
% enhance_imgDir = ["/home/ubuntu/sharedData/YYK/Myenhance/test/KinD_p/"];
% enhance_imgDir = ["/home/ubuntu/sharedData/YYK/Myenhance/test/Zero-DCE/"];
% enhance_imgDir = ["/home/ubuntu/sharedData/YYK/Myenhance/test/PEC_cg/"];
oldPwd = pwd;  

all_under_niqe = [];
all_over_niqe = [];


for nn = 1:length(enhance_imgDir)
disp(enhance_imgDir(nn))
cd(enhance_imgDir(nn));  
x = dir;  
list_Of_enhance_Images = [];  
for i = 1:length(x)
    if x(i).isdir == 0
          list_Of_enhance_Images = [list_Of_enhance_Images; x(i)];  
    end  
end
cd(oldPwd);  

cd(input_imgDir);  
x = dir;  
list_Of_input_Images = [];  
for i = 1:length(x)
    if x(i).isdir == 0
          list_Of_input_Images = [list_Of_input_Images; x(i)];  
    end  
end
cd(oldPwd);  

num = 1;
under_niqe = [];
over_niqe = [];
for i = 1:length(list_Of_enhance_Images)
    enhance_img_path = [list_Of_enhance_Images(i).folder, '/', list_Of_enhance_Images(i).name];
    input_img_path = [list_Of_input_Images(i).folder, '/', list_Of_input_Images(i).name];
    enhance_img = imread(enhance_img_path);
    niqe = computequality(enhance_img,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap,mu_prisparam,cov_prisparam);
    disp(num);
    disp(niqe);
    if mod(num, 5) == 2 || mod(num, 5) == 3
        under_niqe = [under_niqe; niqe];
    end
    if mod(num, 5) == 1 || mod(num, 5) == 4 || mod(num, 5) == 0
        over_niqe = [over_niqe; niqe];
    end
    num = num + 1;
end

all_under_niqe = [all_under_niqe; sum(under_niqe) / length(under_niqe)];
all_over_niqe = [all_over_niqe; sum(over_niqe) / length(over_niqe)];

disp(sum(under_niqe) / length(under_niqe));
disp(sum(over_niqe) / length(over_niqe));

end
disp('all_under_niqe ');
disp(sum(all_under_niqe) / length(all_under_niqe));
disp('all_over_niqe ');
disp(sum(all_over_niqe) / length(all_over_niqe));
disp( (sum(all_under_niqe) + sum(all_over_niqe)) / (length(all_under_niqe) + length(all_over_niqe)) )
