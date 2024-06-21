function [M] = saveSpectrumToTXT(FID,peaknumber,iter,f)
F=fft(FID);
F = F/f;
Size=size(F);
N1=Size(2);
M=zeros(N1,2);
M(:,1)=real(F(1,:));
M(:,2)=imag(F(1,:));
% figure,plot(real(F))
% figure,plot(real(FID))
M=single(M);
end
