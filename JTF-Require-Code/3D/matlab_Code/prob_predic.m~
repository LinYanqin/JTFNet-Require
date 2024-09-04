% % %%%%%%%%%%%%%%%%% temp reconstruction data %%%%%%%%%%%%%%%%%%%%%%%
z_axis = 732;
y_axis = 128;
x_axis = 128;
res_3D = zeros(10,y_axis,z_axis,x_axis);
res_CN = zeros(10,y_axis,x_axis);
max_res_3D = zeros(1,10);
max_res_CN = zeros(1,10);
for i = 1:10
    name = ['../Rec_Data/Rec_mat/resCN3D',num2str(i),'.mat'] ;
    data = load(name);
    data = data.resCN3D;
    max_res_3D(1,i) = max((data(:)));
    res_3D(i,:,:,:) = data/max_res_3D(1,i);
    name = ['../Rec_Data/Rec_mat/resCN',num2str(i),'.mat'] ;
    data = load(name);
    data = data.resCN;
    max_res_CN(1,i) = max(data(:));
    res_CN(i,:,:) = data/max_res_CN(1,i);
end
resCN = reshape(mean(res_CN),[y_axis,x_axis]);
max_f = max((max_res_3D(:)));
res_mean = reshape(mean(res_3D),[y_axis,z_axis,x_axis]);
name = ['../Rec_Data/Rec_mat/ale3D.mat'] ;
data = load(name);
ale = data.ale3D;
name = ['./label_CN3D.mat'];
data = load(name);
labeltemp3D = data.label_CN3D;
name = ['./label_CN.mat'];
data = load(name);
labeltempCN = data.label_CN;
label = labeltemp3D/max(labeltemp3D(:));
label_CN = labeltempCN/max(abs(labeltempCN(:)));
figure,contour(abs(label_CN),10),title('label')
figure,contour(abs(resCN),10),title('rec')
res_mean = permute(res_mean,[1,3,2]);
label = permute(label,[1,3,2]);
res_mean = reshape(res_mean,[y_axis*z_axis,x_axis]);
label = reshape(label,[y_axis*z_axis,x_axis]);
RLNE = norm(res_mean-label)/norm(label)
temp = var(res_3D);
epi = mean(temp(:))
ale = abs(mean(ale(:)))/max_f;
ale = ale/z_axis;
%%%%%%%%%%%%%%%%%%%%   RLNE Uncertain  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

epi_data = load('./epi_paper1.mat');
ale_data = load('./ale_paper1.mat');
rlne_data = load('./rlne_paper1.mat');
epi_sort1 = epi_data.epi_sort1;
ale_sort1 = ale_data.ale_sort1;
rlne_sort = rlne_data.rlne_sort;

%%%%%%%%%%%%%%%%% quality assessment %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
epi = epi;
ale =  ale;
gap_epi = 1e-11;
gap_ale = 1e-7;
upper = 0;
lower = 0;
if epi > 3e-3 || ale > 2e-3
    p_rlne = 0
else
    while lower < 100
        lower = 0;
        gap_epi = gap_epi + 1e-11;
        gap_ale = gap_ale + 1e-7;
        for i = 1: 5000
            if epi_sort1(1,i)>epi-gap_epi && epi_sort1(1,i)<epi+gap_epi && ale_sort1(1,i)>ale-gap_ale && ale_sort1(1,i)<ale+gap_ale
                lower = lower + 1;
            end
        end
    end
    for i = 1:5000
        if epi_sort1(1,i)>epi-gap_epi && epi_sort1(1,i)<epi+gap_epi && ale_sort1(1,i)>ale-gap_ale && ale_sort1(1,i)<ale+gap_ale && rlne_sort(1,i)<=0.38
            upper = upper + 1;
        end
    end
    p_rlne = upper/(lower)
end
