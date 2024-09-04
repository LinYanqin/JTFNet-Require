
name_res1 = ['../Res_data/res1.mat'];
res_data1 = load(name_res1);
res_data1 = res_data1.res;
res_data1 = permute(res_data1,[3,4,2,1]);
name_res2 = ['../Res_data/res2.mat'];
res_data2 = load(name_res2);
res_data2 = res_data2.res;
res_data2 = permute(res_data2,[3,4,2,1]);
for i = 1:10
    res_3D = zeros(128,128,732);
    for k = 1:732
        temp = ifft2(ifftshift(res_data1(:,:,k,i))); 
        res_3D(1:2:end,1:2:end,k) = real(temp);
        res_3D(1:2:end,2:2:end,k) = imag(temp);
        temp = ifft2(ifftshift(res_data2(:,:,k,i))); 
        res_3D(2:2:end,1:2:end,k) = real(temp);
        res_3D(2:2:end,2:2:end,k) = imag(temp);
    end
    name_res = ['../Res_data/res_full',num2str(i),'.mat'];
    save(name_res, 'res_3D');
    
end