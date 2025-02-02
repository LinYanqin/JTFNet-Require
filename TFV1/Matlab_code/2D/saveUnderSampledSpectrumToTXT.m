function [factor,M] = saveUnderSampledSpectrumToTXT( FID,peaknumber,iter)
F=fft(FID);
temp = real(F(:));
factor = max(temp(:));
F=F/factor;
Size=size(F);
N1=Size(2);
M=zeros(N1,2);
M(:,1)=real(F);
M(:,2)=imag(F);
M=single(M);
end
