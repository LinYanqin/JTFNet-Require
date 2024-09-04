import nmrglue as ng
#import mat4py as mt
import pylab
import numpy as np
import scipy.io as sio
# #
dic, data = ng.pipe.read("/data1/ly/demo/gb3/test.ft2")
#
print(data.ndim)
print(data.shape)
print(data.dtype)
#labeltempCN
#labeltemp3D
sio.savemat('/data1/ly/JTF-Require-Code/Matlab_Code/preprocess/GB1.mat',{'data':data})

# dic, data = ng.pipe.read("/data4/ly/4D_SMILE/hmqcnoesyhmqc_CH3.4D/test_all.ft1")
# #
# print(data.ndim)
# print(data.shape)
# print(data.dtype)
# #labeltempCN
# #labeltemp3D
# for i in range(646):
#     temp = data[:,:,:,i]
#     sio.savemat('/data4/ly/4D_SMILE/4D_mat/data4D_%d.mat'%(int(i)),{'data4D':temp})
