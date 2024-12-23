traininput_path = '../../Dataset/2D/train/Input/'; 
trainlabel_path = '../../Dataset/2D/train/Label/';
trainmask_path = '../../Dataset/2D/train/Mask/';
valinput_path = '../../Dataset/2D/val/Input/'; 
vallabel_path = '../../Dataset/2D/val/Label/';
valmask_path = '../../Dataset/2D/val/Mask/';

if ~exist(traininput_path, 'dir')
    mkdir(traininput_path);
end
if ~exist(trainlabel_path, 'dir')
    mkdir(trainlabel_path);
end
if ~exist(trainmask_path, 'dir')
    mkdir(trainmask_path);
end
if ~exist(valinput_path, 'dir')
    mkdir(valinput_path);
end
if ~exist(vallabel_path, 'dir')
    mkdir(vallabel_path);
end
if ~exist(valmask_path, 'dir')
    mkdir(valmask_path);
end
type = 'val'; % input "train" or "val" to generate training set or validation set
load('./mask_pos_num20000.mat');
N1=175;
c = 1;
max_peak = 10;
if strcmp(type, 'train')
    input_data = zeros(40000,1,176,2);
    label_data = zeros(40000,1,176,2);
    mask_data = zeros(40000,1,176);
    fs = 1000;
    for p = 1:max_peak
        for iter=1:4000
            FID1 = 0;
            for i = 1:p 
                Amplitude=0.001+0.999*rand(1); 
                Omega = (fs-100)*rand(1);
                Tao1=0.01+(0.19)*rand(1); 
                temp = generate1DFID_multiplepeaks(Amplitude,Omega,Tao1,N1,fs); 
                FID1 = FID1 + temp;
            end
            rand_num = randperm(20000,1);
            mask_index = mask_pos{1,rand_num};
            mask = zeros(1,176);
            l = length(mask_index);
            for k = 1:1:l
                mask(1,mask_index(1,k)+1) = 1;
            end
            FID2=FID1.*mask;
            [f,input] = saveUnderSampledSpectrumToTXT(FID2,p,iter);
            label = saveSpectrumToTXT(FID1,p,iter,f);
            input_data(c,1,:,1) = input(:,1);
            input_data(c,1,:,2) = input(:,2);
            label_data(c,1,:,1) = label(:,1);
            label_data(c,1,:,2) = label(:,2);
            mask_data(c,1,:) = mask(:);
            c = c + 1
        end
    end
    name_input = fullfile('..', '..', 'Dataset', '2D', 'train', 'input_data.mat');
    name_label = fullfile('..', '..', 'Dataset', '2D', 'train', 'label_data.mat');
    name_mask = fullfile('..', '..', 'Dataset', '2D', 'train', 'mask.mat');
    save(name_input,'input_data');
    save(name_label,'label_data');
    save(name_mask,'mask_data');
elseif strcmp(type, 'val')
    input_data = zeros(4000,1,176,2);
    label_data = zeros(4000,1,176,2);
    mask_data = zeros(4000,1,176);
    fs = 1000;
    for p = 1:max_peak
        for iter=1:400
            FID1 = 0;
            for i = 1:p 
                Amplitude=0.001+0.999*rand(1); 
                Omega = (fs-100)*rand(1);
                Tao1=0.01+(0.19)*rand(1); 
                temp = generate1DFID_multiplepeaks(Amplitude,Omega,Tao1,N1,fs); 
                FID1 = FID1 + temp;
            end
            rand_num = randperm(20000,1);
            mask_index = mask_pos{1,rand_num};
            mask = zeros(1,176);
            l = length(mask_index);
            for k = 1:1:l
                mask(1,mask_index(1,k)+1) = 1;
            end
            FID2=FID1.*mask;
            [f,input] = saveUnderSampledSpectrumToTXT(FID2,p,iter);
            label = saveSpectrumToTXT(FID1,p,iter,f);
            input_data(c,1,:,1) = input(:,1);
            input_data(c,1,:,2) = input(:,2);
            label_data(c,1,:,1) = label(:,1);
            label_data(c,1,:,2) = label(:,2);
            mask_data(c,1,:) = mask(:);
            c = c + 1
        end
    end
    name_input = fullfile('..', '..', 'Dataset', '2D', 'val', 'input_data.mat');
    name_label = fullfile('..', '..', 'Dataset', '2D', 'val', 'label_data.mat');
    name_mask = fullfile('..', '..', 'Dataset', '2D', 'val', 'mask.mat');
    save(name_input,'input_data');
    save(name_label,'label_data');
    save(name_mask,'mask_data');
end

