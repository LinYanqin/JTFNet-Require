% function [St] = generate2DFID_onepeak(Amplitude,Omega1,Tao1,Omega2,Tao2,N1,N2,fs1,fs2)
% n1=0:1:N1-1;
% n2=0:1:N2-1;
% t1=n1/fs1;
% t2=n2/fs2;
% S1=Amplitude*exp(1i*2*pi*Omega1*t1).*exp(-t1/Tao1);
% S1 = fft(S1)';
% I1 = cos(2*pi*Omega2*t2).*real(S1);
% I2 = sin(2*pi*Omega2*t2).*real(S1);
% S = I1 + i*I2;
% S = S.*exp(-t2/Tao2);
% S = fft(S,N2,2);
% St = ifft2(S);
% end
% function [St] = generate2DFID_onepeak(Amplitude,Omega1,Tao1,Omega2,Tao2,N1,N2,fs1,fs2)
% n1=0:1:N1-1;
% n2=0:1:N2-1;
% t1=n1/fs1;
% t2=n2/fs2;
% S1=Amplitude*exp(1i*2*pi*Omega1*t1).*exp(-t1/Tao1);
% S1 = fft(S1)';
% S = exp(1i*2*pi*Omega2*t2).*real(S1);
% S = S.*exp(-t2/Tao2);
% S = fft(S,N2,2);
% St = ifft2(S);
% end
% % function [S] = generate2DFID_onepeak(Amplitude,Omega1,Tao1,Omega2,Tao2,N1,N2,fs1,fs2)
% n1=0:1:N1-1;
% n2=0:1:N2-1;
% t1=n1/fs1;
% t2=n2/fs2;
% S1=Amplitude*exp(1i*2*pi*Omega1*t1).*exp(-t1/Tao1);
% S2 = exp(1i*2*pi*Omega2*t2).*exp(-t2/Tao2);
% S = S1'.*S2;
% end
function [fid] = generate2DFID_onepeak(Amplitude,Omega1,Tao1,Omega2,Tao2,N1,N2,fs1,fs2,offdata,enddata)
t1=0:1/fs1:(N1-1)/fs1;
t2=0:1/fs2:(N2-1)/fs2;
S1=Amplitude*exp(1i*2*pi*Omega1*t1).*exp(-t1/Tao1);
S1 = applySineWindow(S1,offdata,enddata,1,0.5);
S2 = exp(1i*2*pi*Omega2*t2).*exp(-t2/Tao2);
S2 = applySineWindow(S2,offdata,enddata,1,0.5);
R1 = real(S1);
I1 = imag(S1);
R2 = real(S2);
I2 = imag(S2);
fid = zeros(128,128);
R1R2 = R1.'*R2;
R1I2 = R1.'*I2;
I1R2 = I1.'*R2;
I1I2 = I1.'*I2;
fid(1:2:end,1:2:end) = R1R2;
fid(1:2:end,2:2:end) = R1I2;
fid(2:2:end,1:2:end) = I1R2;
fid(2:2:end,2:2:end) = I1I2;
end
% 
% 
