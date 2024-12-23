function [FID] = generate2DFID_multiplepeaks(Amplitude,Omega1,Tao1,Omega2,Tao2,N1,N2,fs1,fs2,off,enddata)
peak_number=length(Amplitude);
for iter=1:peak_number
    if iter==1
        [FID] =generate2DFID_onepeak(Amplitude(iter),Omega1(iter),Tao1(iter),Omega2(iter),Tao2(iter),N1,N2,fs1,fs2,off,enddata);
    else
        [FID1] =generate2DFID_onepeak(Amplitude(iter),Omega1(iter),Tao1(iter),Omega2(iter),Tao2(iter),N1,N2,fs1,fs2,off,enddata);
        FID=FID1 + FID;
    end
end
end