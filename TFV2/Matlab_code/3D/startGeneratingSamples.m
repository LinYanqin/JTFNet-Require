traininput_path = '../../Dataset/3D/train/Input/'; 
trainlabel_path = '../../Dataset/3D/train/Label/';
trainmask_path = '../../Dataset/3D/train/Mask/';
valinput_path = '../../Dataset/3D/val/Input/'; 
vallabel_path = '../../Dataset/3D/val/Label/';
valmask_path = '../../Dataset/3D/val/Mask/';

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
load('./mask_pos_20000.mat');
MaxPeaks=20;
N1=64; 
N2=64; 
NumOfSamples=1000;
k = 1;
num = 1;
for m = 1:1:1
    k=1;
    for i=1:1:MaxPeaks
        for iter=1:NumOfSamples 
             Amplitude=0.001+(1-0.001)*rand(1,i);
             Tao1=0.001+(0.019)*rand(1,i);
             Tao2=0.001+(0.019)*rand(1,i);
             fs1=4000; 
             Omega1=fs1*(0.05 + (0.95-0.05)*rand(1,i));
             fs2=4000;
             Omega2=fs2*(0.05 + (0.95-0.05)*rand(1,i)); 
             
             rand_off = randperm(4,1);
             rand_end = randperm(4,1);
             off_data = [0.3,0.4,0.5,0.6];
             enddata = [0.85,0.9,0.95,1];
             FID1=generate2DFID_multiplepeaks(Amplitude,Omega1,Tao1,Omega2,Tao2,N1,N2,fs1,fs2,off_data(rand_off),enddata(rand_end));
             R1R2 = FID1(1:2:end,1:2:end);
             R1I2 = FID1(1:2:end,2:2:end); 
             I1R2 = FID1(2:2:end,1:2:end);
             I1I2 = FID1(2:2:end,2:2:end);

             FID_1 = R1R2 + j*R1I2;
             FID_2 = I1R2 + j*I1I2;

             rand_num = randperm(20000,1);
             mask_index = mask_pos_3d{1,rand_num};
             mask = zeros(N1,N2);
             l = length(mask_index);
             for h = 1:1:l
                   mask(mask_index(h,1)+1,mask_index(h,2)+1) = 1;
             end
             FID_3 = FID_1.*mask;
             FID_4 = FID_2.*mask;
             name1 = ['../../Dataset/3D/train/Input/input',num2str(k),'.mat'];
             name2 = ['../../Dataset/3D/train/Label/label',num2str(k),'.mat'];
             name3 = ['../../Dataset/3D/train/Mask/mask',num2str(k),'.mat'];
             [f,under] = saveUnderSampledSpectrumToTXT(FID_3,i,iter);
             [full] = saveSpectrumToTXT(FID_1,i,iter,f);
             save(name1,'under');
             save(name2,'full');
             save(name3,'mask');
             [f,under] = saveUnderSampledSpectrumToTXT(FID_4,i,iter);
             full = saveSpectrumToTXT(FID_2,i,iter,f);
             name4 = ['../../Dataset/3D/train/Input/input',num2str(k+1),'.mat'];
             name5 = ['../../Dataset/3D/train/Label/label',num2str(k+1),'.mat'];
             name6 = ['../../Dataset/3D/train/Mask/mask',num2str(k+1),'.mat'];
             save(name4,'under');
             save(name5,'full');
             save(name6,'mask');
             k = k + 2
             num = num + 1;
        end
    end
end
NumOfSamples=100;
k = 1;
num = 1;
for m = 1:1:1
    k=1;
    for i=1:1:MaxPeaks
        for iter=1:NumOfSamples 
             Amplitude=0.001+(1-0.001)*rand(1,i);
             Tao1=0.001+(0.019)*rand(1,i);
             Tao2=0.001+(0.019)*rand(1,i);
             fs1=4000; 
             Omega1=fs1*(0.05 + (0.95-0.05)*rand(1,i));
             fs2=4000;
             Omega2=fs2*(0.05 + (0.95-0.05)*rand(1,i)); 
             rand_off = randperm(4,1);
             rand_end = randperm(4,1);
             off_data = [0.3,0.4,0.5,0.6];
             enddata = [0.85,0.9,0.95,1];
             FID1=generate2DFID_multiplepeaks(Amplitude,Omega1,Tao1,Omega2,Tao2,N1,N2,fs1,fs2,off_data(rand_off),enddata(rand_end));
             R1R2 = FID1(1:2:end,1:2:end);
             R1I2 = FID1(1:2:end,2:2:end); 
             I1R2 = FID1(2:2:end,1:2:end);
             I1I2 = FID1(2:2:end,2:2:end);

             FID_1 = R1R2 + j*R1I2;
             FID_2 = I1R2 + j*I1I2;

             rand_num = randperm(20000,1);
             mask_index = mask_pos_3d{1,rand_num};
             mask = zeros(N1,N2);
             l = length(mask_index);
             for h = 1:1:l
                   mask(mask_index(h,1)+1,mask_index(h,2)+1) = 1;
             end
             FID_3 = FID_1.*mask;
             FID_4 = FID_2.*mask;
             name1 = ['../../Dataset/3D/val/Input/input',num2str(k),'.mat'];
             name2 = ['../../Dataset/3D/val/Label/label',num2str(k),'.mat'];
             name3 = ['../../Dataset/3D/val/Mask/mask',num2str(k),'.mat'];
             [f,under] = saveUnderSampledSpectrumToTXT(FID_3,i,iter);
             [full] = saveSpectrumToTXT(FID_1,i,iter,f);
             save(name1,'under');
             save(name2,'full');
             save(name3,'mask');
             [f,under] = saveUnderSampledSpectrumToTXT(FID_4,i,iter);
             full = saveSpectrumToTXT(FID_2,i,iter,f);
             name4 = ['../../Dataset/3D/val/Input/input',num2str(k+1),'.mat'];
             name5 = ['../../Dataset/3D/val/Label/label',num2str(k+1),'.mat'];
             name6 = ['../../Dataset/3D/val/Mask/mask',num2str(k+1),'.mat'];
             save(name4,'under');
             save(name5,'full');
             save(name6,'mask');
             k = k + 2
             num = num + 1;
        end
    end
end





