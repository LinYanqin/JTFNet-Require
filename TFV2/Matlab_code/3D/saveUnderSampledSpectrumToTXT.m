function [factor, M] = saveUnderSampledSpectrumToTXT( FID,peaknumber,iter)
F=fftshift(fft2(FID));
temp = abs(real(F(:)));
factor = max(temp(:));
F=F/factor;
Size=size(F);
N1=Size(1);
N2=Size(2);
M=zeros(N1,N2,2);
M(:,:,1)=real(F);
M(:,:,2)=imag(F);
% figure,contour(real(F),10);
% figure,mesh(real(F));
M=single(M);
end
