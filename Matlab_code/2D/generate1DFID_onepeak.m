function [S] = generate1DFID_onepeak(Amplitude,Omega1,Tao1,N1,fs1)  
n1=0:1:N1;
t1=n1/fs1;
S=Amplitude*exp(1i*2*pi*Omega1*t1).*exp(-t1/Tao1);
end


