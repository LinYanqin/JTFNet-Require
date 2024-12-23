clear
clc
%%%%%%%%%%%%%%%%%%% load quality space %%%%%%%%%%%%%%%%%%%%
load('Ale_3D_NC.mat');
load('Epi_3D_NC.mat');
load('RLNE_3D_NC.mat');
% % %%%%%%%%%%%%%%%%% temp reconstruction data %%%%%%%%%%%%%%%%%%%%%%%
num_iter = 10;
rec_path = "../../NMRPipe_code/3D/res_mat/";
direct_dim = 732;
indirect_y = 128;
indirect_z = 128;
full_sampled = 1;%0/1
z_axis = direct_dim;
y_axis = indirect_y;
x_axis = indirect_z;
res_3D = zeros(num_iter,y_axis,z_axis,x_axis);
res_CN = zeros(num_iter,y_axis,x_axis);
max_res_3D = zeros(1,num_iter);
max_res_CN = zeros(1,num_iter);
for i = 1:num_iter
    name = fullfile(rec_path, ['res3D',num2str(i),'.mat']);
    data = load(name);
    data = data.res3D;
    max_res_3D(1,i) = max((data(:)));
    res_3D(i,:,:,:) = data/max_res_3D(1,i);
    name = fullfile(rec_path,[ 'res_proj',num2str(i),'.mat']) ;
    data = load(name);
    data = data.res_proj;
    max_res_CN(1,i) = max(data(:));
    res_CN(i,:,:) = data/max_res_CN(1,i);
end
resCN = reshape(mean(res_CN),[y_axis,x_axis]);
max_f = max((max_res_3D(:)));
res_mean = reshape(mean(res_3D),[y_axis,z_axis,x_axis]);
name = fullfile(rec_path,['ale3D.mat']) ;
data = load(name);
ale = data.ale3D;
if full_sampled == 1
    name = fullfile(rec_path,['labeltemp3D.mat']);
    data = load(name);
    labeltemp3D = data.labeltemp3D;
    name = fullfile(rec_path,['labeltemp_proj.mat']);
    data = load(name);
    labeltempCN = data.labeltemp_proj;
    label = labeltemp3D/max(abs(labeltemp3D(:)));
    label_CN = labeltempCN/max(abs(labeltempCN(:)));
    figure,contour(real(label_CN),14),title('label')
    res_mean = permute(res_mean,[1,3,2]);
    label = permute(label,[1,3,2]);
    RLNE = norm(res_mean(:)-label(:))/norm(label(:))
end
figure,contour(real(resCN),14),title('rec')
temp = var(res_3D);
epi = mean(temp(:));
ale = abs(mean(ale(:)))/max_f;

%%%%%%%%%%%%%%%%% quality assessment %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gap_epi = 1e-11;
gap_ale = 1e-7;
upper = 0;
lower = 0;
while lower < 100
    lower = 0;
    gap_epi = gap_epi + 1e-11;
    gap_ale = gap_ale + 1e-7;
    for i = 1: 5000
        if Epi_3D_paper(1,i)>epi-gap_epi && Epi_3D_paper(1,i)<epi+gap_epi && Ale_3D_paper(1,i)>ale-gap_ale && Ale_3D_paper(1,i)<ale+gap_ale
            lower = lower + 1;
        end
    end
end
for i = 1:5000
    if Epi_3D_paper(1,i)>epi-gap_epi && Epi_3D_paper(1,i)<epi+gap_epi && Ale_3D_paper(1,i)>ale-gap_ale && Ale_3D_paper(1,i)<ale+gap_ale && RLNE_3D_paper(1,i)<=0.55
        upper = upper + 1;
    end
end
REQUIRER = upper/(lower);
disp(['REQUIRER: ', num2str(REQUIRER*100),'%']);

