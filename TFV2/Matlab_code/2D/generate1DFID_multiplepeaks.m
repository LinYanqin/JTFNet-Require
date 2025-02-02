function [FID] = generate1DFID_multiplepeaks(Amplitude,Omega1,Tao1,N1,fs1)
peak_number=length(Amplitude);
for iter=1:peak_number
    if iter==1
        FID=generate1DFID_onepeak(Amplitude(iter),Omega1(iter),Tao1(iter),N1,fs1);
    else
        FID=FID+generate1DFID_onepeak(Amplitude(iter),Omega1(iter),Tao1(iter),N1,fs1);
    end
end
end