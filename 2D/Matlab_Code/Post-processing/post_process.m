res_data = zeros(10,601,176);
ale_data = zeros(10,601,176);
for k = 1:10
    name_res = ['./rec_temp/res_temp_',num2str(k-1),'.mat'];
    name_ale = ['./rec_temp/ale_temp_',num2str(k-1),'.mat'];
    data = load(name_res);
    data = data.F;
    res = zeros(601,176);
    res(:,:) = data(:,1,:);
    res = factor'.*res; 
    max_res = max(real(res(:)));
    res = res/max_res;
    res_data(k,:,:) = res(:,:);
    data = load(name_ale);
    data = data.V;
    ale = zeros(601,176);
    ale(:,:) = data(:,1,:);
    ale = factor'.*ale;
    ale = ale/max_res;
    ale_data(k,:,:) = abs(ale(:,:));
end
label_f = label_gb1/max(real(label_gb1(:)));
res = mean(res_data);
res = permute(res,[2,3,1]);
figure,contour(real(res),10)
epi_data = var(res_data);
epi_data = permute(epi_data,[2,3,1]);
epi_data = mean(epi_data(:));
ale_data = mean(ale_data(:));
RLNE_reflex = norm(res-label_f)/norm(label_f)
%%%%%%%%%%%%%%%%% Requirer %%%%%%%%%%%%%%%%%%%%%

load('./Ale_2D_paper');
load('./Epi_2D_paper');
load('./RLNE_2D_paper');
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
        if Epi_2D_paper(1,i)>epi-gap_epi && Epi_2D_paper(1,i)<epi+gap_epi && Ale_2D_paper(1,i)>ale-gap_ale && Ale_2D_paper(1,i)<ale+gap_ale && RLNE_2D_paper(1,i)<=0.26
            upper = upper + 1;
        end
end
Require = upper/(lower)