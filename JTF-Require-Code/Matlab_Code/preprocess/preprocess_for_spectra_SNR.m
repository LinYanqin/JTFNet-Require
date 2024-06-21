%%%%%%%%%%%% load data %%%%%%%%%%%%%
name = ['./GB1.mat'];
protein_data  = load(name);
label_gb1 = fliplr(protein_data.data);
[y_axis,x_axis] = size(label_gb1);
figure,contour(real(label_gb1),10)
%%%%%%%%%%%% Normalized data %%%%%%%%%%%
label_gb1 = label_gb1/max(real(label_gb1(:)));
t_label = ifft2(label_gb1);
langda=1e-10;
noise1=sqrt(langda)*randn(601,176)+1i*sqrt(langda)*randn(601,176);
t_label = t_label + noise1;
t_label = fft(t_label,y_axis,1);
figure,plot(real(t_label(1,:)))
labelGB1 = fft(t_label,x_axis,2);
label_SNR = labelGB1/max(real(labelGB1(:)));
area = label_SNR(501:601,166:176);
SNR = 1/std(area(:))
%%%%%%%%%%%% generate mask for NUS %%%%%%%%%%%%
mask_index = textread('./mask_temp_10p.txt');
mask = zeros(1,x_axis);
l = length(mask_index);
for k = 1:1:l
    mask(1,mask_index(k,1)+1) = 1;
end
undersample = t_label.*mask;
%%%%%%%%%%% generate_inputs_for_JTF-Net %%%%%%%%%%%%
input = zeros(y_axis,1,x_axis,2);
label = zeros(y_axis,1,x_axis,2);
factor = zeros(1,y_axis);
mask_data = zeros(y_axis,1,x_axis);
for i = 1:y_axis
    FID1 = t_label(i,:);
    FID2 = undersample(i,:);
    [f,inputtemp] = saveUnderSampledSpectrumToTXT(FID2,1,1);
    labeltemp = saveSpectrumToTXT(FID1,1,1,f);
    input(i,1,:,1) = inputtemp(:,1);
    input(i,1,:,2) = inputtemp(:,2);
    label(i,1,:,1) = labeltemp(:,1);
    label(i,1,:,2) = labeltemp(:,2);
    factor(1,i) = f;
    mask_data(i,1,:) = mask;
end
name_input = ['./GB1/input_data.mat'];
name_label = ['./GB1/label_data.mat'];
name_mask = ['./GB1/mask.mat'];
name_factor = ['./GB1/factor.mat'];
save(name_input,'input');
save(name_label,'label');
save(name_mask,'mask_data');
save(name_factor,'factor');

