function [factor,M] = saveUnderSampledSpectrumToTXT( FID,peaknumber,iter)
F=fft(FID);
% real_temp = max(abs(real(F)));
% imag_temp = max(abs(imag(F)));
temp = real(F(:));
factor = max(temp(:));
F=F/factor;
Size=size(F);
N1=Size(2);
M=zeros(N1,2);
M(:,1)=real(F);
M(:,2)=imag(F);
% figure,plot(real(F))
% figure,plot(real(FID))
% figure,plot(real(ifft(F)))
M=single(M);
% dlmwrite(strcat(datapath,FileName), M,'delimiter' , ' ', 'newline', 'pc');
end
