%%%%%%%%%%%% load data %%%%%%%%%%%%%
name = ['./protein.mat'];
protein_data  = load(name);
label_protein = fliplr(protein_data.protein);
[y_axis,x_axis] = size(label_protein);
figure,contour(real(label_protein),10)
%%%%%%%%%%%% Normalized data %%%%%%%%%%%
label_protein = label_protein/max(real(label_protein(:)));
t_label = ifft(label_protein,x_axis,2);
figure,plot(real(t_label(1,:)))
%%%%%%%%%%%% generate mask for NUS %%%%%%%%%%%%
mask_index = textread('./mask_temp.txt');
mask = zeros(1,x_axis);
l = length(mask_index);
for k = 1:1:l
    mask(1,mask_index(k,1)+1) = 1;
end
undersample = t_label;
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
    factor(1,i) = f;
    mask_data(i,1,:) = mask;
end
name_input = ['./protein/input_data.mat'];
name_mask = ['./protein/mask.mat'];
name_factor = ['./protein/factor.mat'];
save(name_input,'input');
save(name_mask,'mask_data');
save(name_factor,'factor');

