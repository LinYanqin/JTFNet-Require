y_axis = 64;
x_axis = 64;
z_axis = 732;
index = textread('./mask3D.txt');
num = length(index);
mask= zeros(64,64);
for i = 1:num
    mask((index(i,1)+1),(index(i,2)+1)) = 1; %Generate mask
end

name = ['./GB1_3D_t.mat'];
data = load(name);
data = data.GB1_3D_t;
fid = data;
R1R2 = fid(1:2:end,1:2:end,:);
R1I2 = fid(1:2:end,2:2:end,:);
I1R2 = fid(2:2:end,1:2:end,:);
I1I2 = fid(2:2:end,2:2:end,:);
FID_real1 = R1R2+j*R1I2;
FID_real2 = I1R2+j*I1I2; %Recombine data
temp = fft(FID_real1,64,1);
temp = fft(temp,64,2);
temp = temp/max(real(temp(:)));
area = temp(50:60,50:60,1:20);
SNR = 1/std(area(:))
input_real1 = zeros(z_axis,y_axis,x_axis,2);
input_real2 = zeros(z_axis,y_axis,x_axis,2);
noise_label = zeros(y_axis*2,x_axis*2,z_axis);
factor1 = zeros(1,z_axis);
factor2 = zeros(1,z_axis);
noise_label(1:2:end,1:2:end,:) = real(FID_real1);
noise_label(1:2:end,2:2:end,:) = imag(FID_real1);
noise_label(2:2:end,1:2:end,:) = real(FID_real2);
noise_label(2:2:end,2:2:end,:) = imag(FID_real2);
for i = 1:z_axis
    FID1 = FID_real1(:,:,i);
    FID1_under = FID1;
    FID2 = FID_real2(:,:,i);
    FID2_under = FID2;
    [f,under] = saveUnderSampledSpectrumToTXT(FID1_under,i,1);
    factor1(1,i) = f;
    input_real1(i,:,:,:) = under;
    [f,under] = saveUnderSampledSpectrumToTXT(FID2_under,i,1);
    factor2(1,i) = f;
    input_real2(i,:,:,:) = under;
end
noiselabel_name = ['../Processed_Data/noise_label.mat'];
label_name = ['../Processed_Data/label_3D.mat'];
input_name1 = ['../Processed_Data/inputreal',num2str(1),'.mat'];
input_name2 = ['../Processed_Data/inputreal',num2str(2),'.mat'];
factor_name1 = ['../Processed_Data/factor',num2str(1),'.mat'];
factor_name2 = ['../Processed_Data/factor',num2str(2),'.mat'];
mask_name = ['../Processed_Data/mask3D.mat'];
    
save(input_name1,'input_real1');
save(input_name2,'input_real2');
save(factor_name1,'factor1');
save(factor_name2,'factor2');
save(mask_name,'mask');
save(noiselabel_name,'noise_label');
save(label_name,'fid');