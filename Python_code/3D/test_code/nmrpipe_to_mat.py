import nmrglue as ng
import os
import scipy.io as sio

rec_path = "../../../NMRPipe_code/3D/"
num_iter = 10
#
folder_temp = rec_path + 'res_mat'
if not os.path.exists(folder_temp):
    os.makedirs(folder_temp)
    print(f"Folder {folder_temp} is not exit，begin to bulit this folder")
else:
    print(f"Folder {folder_temp} exit")
for i in range(num_iter):
    dic, data = ng.pipe.read(rec_path + "res_nmrpipe/res_proj%d.dat"%int(i+1))
    #
    print(data.ndim)
    print(data.shape)
    print(data.dtype)
    #
    #
    sio.savemat(rec_path + 'res_mat/res_proj%d.mat'%int(i+1),{'res_proj':data})

############################### 3D output ####################################
for i in range(num_iter):
    dic, data = ng.pipe.read(rec_path + "res_nmrpipe/res3D%d.dat"%int(i+1))
    #
    print(data.ndim)
    print(data.shape)
    print(data.dtype)
    #
    #
    sio.savemat(rec_path + 'res_mat/res3D%d.mat'%int(i+1),{'res3D':data})
dic, data = ng.pipe.read(rec_path + "res_nmrpipe/ale3D.dat")
    #
print(data.ndim)
print(data.shape)
print(data.dtype)
    #
    #
sio.savemat(rec_path + 'res_mat/ale3D.mat',{'ale3D':data})

type = input("Convert the full sampled spectrum to mat or not?(Y/N)")
if type == "Y":
    proj_name = input("Please input the path of the projection file:")
    label_3D = input("Please input the path of the full sampled 3D spectrum:")
    dic, data = ng.pipe.read(proj_name)
        #
    print(data.ndim)
    print(data.shape)
    print(data.dtype)
        #
        #
    sio.savemat(rec_path+'res_mat/labeltemp_proj.mat',{'labeltemp_proj':data})

    dic, data = ng.pipe.read(label_3D)
        #
    print(data.ndim)
    print(data.shape)
    print(data.dtype)
        #
        #
    sio.savemat(rec_path+'res_mat/labeltemp3D.mat',{'labeltemp3D':data})

