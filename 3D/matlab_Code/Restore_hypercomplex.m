x_axis = 64;
y_axis = 64;
z_axis = 732;
ale_3D = zeros(y_axis*2,x_axis*2,z_axis);

name_ale1 = ['../Rec_Data/Rec_mat/ale1.mat'];
name_res1 = ['../Rec_Data/Rec_mat/res1.mat'];
ale_data1 = load(name_ale1);
res_data1 = load(name_res1);
ale_data1 = ale_data1.ale;
res_data1 = res_data1.res;
ale_data1 = permute(ale_data1,[2,3,1]);
res_data1 = permute(res_data1,[3,4,2,1]);
name_ale2 = ['../Rec_Data/Rec_mat/ale2.mat'];
name_res2 = ['../Rec_Data/Rec_mat/res2.mat'];
ale_data2 = load(name_ale2);
res_data2 = load(name_res2);
ale_data2 = ale_data2.ale;
res_data2 = res_data2.res;
ale_data2 = permute(ale_data2,[2,3,1]);
res_data2 = permute(res_data2,[3,4,2,1]);
for i = 1:10
    res_3D = zeros(y_axis*2,x_axis*2,z_axis);
    for k = 1:z_axis
        temp = ifft2(ifftshift(res_data1(:,:,k,i))); 
        res_3D(1:2:end,1:2:end,k) = real(temp);
        res_3D(1:2:end,2:2:end,k) = imag(temp);
        temp = ifft2(ifftshift(res_data2(:,:,k,i))); 
        res_3D(2:2:end,1:2:end,k) = real(temp);
        res_3D(2:2:end,2:2:end,k) = imag(temp);
    end
    name_res = ['../Rec_Data/Rec_mat/res_full',num2str(i),'.mat'];
    save(name_res, 'res_3D');
end
for k = 1:z_axis
    temp = ifft2(ifftshift(ale_data1(:,:,k))); 
    ale_3D(1:2:end,1:2:end,k) = real(temp);
    ale_3D(1:2:end,2:2:end,k) = imag(temp);
    temp = ifft2(ifftshift(ale_data2(:,:,k))); 
    ale_3D(2:2:end,1:2:end,k) = real(temp);
    ale_3D(2:2:end,2:2:end,k) = imag(temp);
end
name_ale = ['../Rec_Data/Rec_mat/ale_full.mat'];
save(name_ale, 'ale_3D');