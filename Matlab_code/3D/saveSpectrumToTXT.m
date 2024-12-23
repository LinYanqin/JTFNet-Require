function [M] = saveSpectrumToTXT(FID,peaknumber,iter,f)
F=fftshift(fft2(FID));
F = F/f;
% figure,contour(real(F),10);
% figure,mesh(real(F));
Size=size(F);
N1=Size(1);
N2=Size(2);
M=zeros(N1,N2,2);
M(:,:,1)=real(F);
M(:,:,2)=imag(F);
M=single(M);
end