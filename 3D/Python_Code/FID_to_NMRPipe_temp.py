import nmrglue as ng
#import mat4py as mt
import pylab
import numpy as np
import scipy.io as sio


dic, data = ng.pipe.read("/data1/ly/3D_REC/ok.ft")

print(data.ndim)
print(data.shape)
print(data.dtype)

M = sio.loadmat('/data4/ly/3D/real_temp/res_data/ale_full.mat')
Data = M['ale_3D']
Data = np.array(Data,dtype='float32')
ng.pipe.write("/data4/ly/3D/real_temp/res_nmrpipe/ale_full.dat", dic, Data, overwrite=True)

for i in range(10):
    M = sio.loadmat('/data4/ly/3D/real_temp/res_data/res_full%d.mat'%int(i+1))
    Data = M['res_3D']
    Data = np.array(Data, dtype='float32')
    ng.pipe.write("/data4/ly/3D/real_temp/res_nmrpipe/res_full%d.dat"%int(i+1), dic, Data, overwrite=True)