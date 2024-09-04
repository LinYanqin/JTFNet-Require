import nmrglue as ng
#import mat4py as mt
import pylab
import numpy as np
import scipy.io as sio
#
for i in range(10):
    dic, data = ng.pipe.read("/data4/ly/3D/real_temp/res_nmrpipe/resCN%d.dat"%int(i+1))
    #
    print(data.ndim)
    print(data.shape)
    print(data.dtype)
    #
    #
    sio.savemat('/data4/ly/3D/real_temp/res_mat/resCN%d.mat'%int(i+1),{'resCN':data})
# dic, data = ng.pipe.read("/data4/ly/3D/real_temp/res_nmrpipe/aleCN.dat")
#     #
# print(data.ndim)
# print(data.shape)
# print(data.dtype)
#     #
#     #
# sio.savemat('/data4/ly/3D/real_temp/res_mat/aleCN.mat',{'aleCN':data})
############################### 3D output ####################################
for i in range(10):
    dic, data = ng.pipe.read("/data4/ly/3D/real_temp/res_nmrpipe/resCN3D%d.dat"%int(i+1))
    #
    print(data.ndim)
    print(data.shape)
    print(data.dtype)
    #
    #
    sio.savemat('/data4/ly/3D/real_temp/res_mat/resCN3D%d.mat'%int(i+1),{'resCN3D':data})
dic, data = ng.pipe.read("/data4/ly/3D/real_temp/res_nmrpipe/ale3D.dat")
    #
print(data.ndim)
print(data.shape)
print(data.dtype)
    #
    #
sio.savemat('/data4/ly/3D/real_temp/res_mat/ale3D.mat',{'ale3D':data})

dic, data = ng.pipe.read("/data3/ldb/myfile/bmrb/ftp/pub/bmrb/timedomain/bmr15999/nesgBtR244_bmrb15999/btr244_hncacb_4_14_08.fid/labeltempCN.dat")
    #
print(data.ndim)
print(data.shape)
print(data.dtype)
    #
    #
sio.savemat('/data4/ly/3D/real_temp/res_mat/labeltempCN.mat',{'labeltempCN':data})

dic, data = ng.pipe.read("/data3/ldb/myfile/bmrb/ftp/pub/bmrb/timedomain/bmr15999/nesgBtR244_bmrb15999/btr244_hncacb_4_14_08.fid/fidtemp3D.ft3")
    #
print(data.ndim)
print(data.shape)
print(data.dtype)
    #
    #
sio.savemat('/data4/ly/3D/real_temp/res_mat/labeltemp3D.mat',{'labeltemp3D':data})
