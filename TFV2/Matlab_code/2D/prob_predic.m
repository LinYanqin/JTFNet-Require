clear
clc
%%%%%%%%%%%%%%%%%%%   load quality space  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('./Ale_2D_NC.mat');
load('./Epi_2D_NC.mat');
load('RLNE_2D_NC.mat');
%%%%%%%%%%%%%%%%%%%%%%%%%%% load reconstructed results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y_axis =601;
x_axis =176;
num_rec = 10;
full_sampled = 1; % 0/1
res_data = zeros(num_rec,y_axis,x_axis);
ale_data = zeros(num_rec,y_axis,x_axis);
for k = 1:num_rec
    name_res = ['../../Python_code/2D/test_code/temp_data/rec_temp/res_temp_',num2str(k-1),'.mat'];
    name_ale = ['../../Python_code/2D/test_code/temp_data/rec_temp/ale_temp_',num2str(k-1),'.mat'];
    load('../../Python_code/2D/test_code/temp_data/input/factor.mat')
    data = load(name_res);
    data = data.F;
    res = zeros(y_axis,x_axis);
    res(:,:) = data(:,1,:);
    res = factor'.*res;
    max_res = max(real(res(:)));
    res = res/max_res;
    res_data(k,:,:) = res(:,:);
    data = load(name_ale);
    data = data.V;
    ale = zeros(y_axis,x_axis);
    ale(:,:) = data(:,1,:);
    ale = factor'.*ale;
    ale = ale/max_res;
    ale_data(k,:,:) = abs(ale(:,:));
end

res = mean(res_data);
res = permute(res,[2,3,1]);
res = res/max(real(res(:)));
if full_sampled == 1
    load('../../Python_code/2D/test_code/label_path/label2D.mat');
    labelf = label2D/max(real(label2D(:)));
    RLNE = norm(res(:)-labelf(:))/norm((labelf(:)))
    figure,contour(real(labelf),9),title('label');
end
figure,contour(real(res),9),title('rec');
epi_data = var(res_data);
epi_data = permute(epi_data,[2,3,1]);
epi_data = mean(epi_data(:));
ale_data = mean(ale_data(:));



epi = epi_data;
ale =  ale_data;
gap_epi = 1e-11;
gap_ale = 1e-7;

upper = 0;
lower = 0;
while lower < 100
        lower = 0;
        gap_epi = gap_epi + 1e-11;
        gap_ale = gap_ale + 1e-7;
        for i = 1: 10000
            if Epi_2D_paper(1,i)>epi-gap_epi && Epi_2D_paper(1,i)<epi+gap_epi && Ale_2D_paper(1,i)>ale-gap_ale && Ale_2D_paper(1,i)<ale+gap_ale
                lower = lower + 1;
            end
        end
end
for i = 1:10000
        if Epi_2D_paper(1,i)>epi-gap_epi && Epi_2D_paper(1,i)<epi+gap_epi && Ale_2D_paper(1,i)>ale-gap_ale && Ale_2D_paper(1,i)<ale+gap_ale && RLNE_2D_paper(1,i)<=0.35
            upper = upper + 1;
        end
end
REQUIRER = upper/(lower);
disp(['REQUIRER: ', num2str(REQUIRER*100),'%']);


